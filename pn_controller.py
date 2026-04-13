"""Proportional Navigation guidance for the drone intercept game.

Uses only camera (64 px, 90-deg FOV) and IMU (gyro_z, accel_fwd,
accel_lat) -- no direct access to game state.  Derives target bearing
from pixel centroid, LOS rate from bearing change + gyro, and range
from apparent angular size.  A pursuit + PN hybrid steers the heading
while a separate thrust law manages approach and gravity compensation.

The interceptor stays grounded until the target enters its camera FOV.

Two implementations:
  PNController     -- single-env, used by the game Autopilot & enjoy.py
  BatchPNController -- vectorized over N envs, used by tune_pn.py
"""

import json
import math
import os

import numpy as np
from dataclasses import dataclass, asdict, fields

CAM_WIDTH = 64
CAM_FOV = math.pi / 2   # 90 degrees
TARGET_RADIUS = 18.0     # world pixels (must match env/physics)

_PIXEL_IDX = np.arange(CAM_WIDTH, dtype=np.float32)
_CAM_CENTRE = (CAM_WIDTH - 1) / 2.0      # 31.5
_MIN_TOTAL = 0.05                          # detection threshold


@dataclass
class PNParams:
    """Tunable parameters for the PN guidance controller."""
    # -- Guidance --
    N: float = 3.5              # navigation constant (PN gain on LOS rate)
    K_bearing: float = 4.0      # pursuit gain (steer nose toward target)
    K_rate: float = 0.12        # inner-loop gain: omega-error -> differential

    # -- Thrust --
    thrust_approach: float = 0.85   # combined thrust when facing target
    thrust_min: float = 0.30        # floor thrust (gravity comp / search)
    K_accel: float = 0.0            # boost thrust when accel_fwd < hover_ref
                                    # (IMU-based gravity compensation)

    # -- Launch gate --
    launch_min_brightness: float = 0.08  # peak camera value to trigger launch

    # -- Search --
    search_yaw: float = 0.20         # differential while scanning
    prescan_thrust: float = 0.0      # hover thrust for pre-launch scan (0 = stay grounded)
    prescan_yaw: float = 0.0         # differential for pre-launch scan rotation


def load_params(path="pn_params.json"):
    """Load PN parameters from JSON, falling back to defaults."""
    if not os.path.exists(path):
        return PNParams()
    with open(path) as f:
        d = json.load(f)
    valid = {fi.name for fi in fields(PNParams)}
    return PNParams(**{k: d[k] for k in d if k in valid})


def save_params(params, path="pn_params.json"):
    with open(path, "w") as f:
        json.dump(asdict(params), f, indent=2)


# =====================================================================
# Single-environment controller (game / enjoy)
# =====================================================================

class PNController:
    """Camera+IMU proportional-navigation guidance for one environment."""

    def __init__(self, params=None, dt=1.0 / 60.0):
        self.params = params or PNParams()
        self.dt = dt
        self.reset()

    def reset(self):
        self.prev_bearing = 0.0
        self.prev_visible = False
        self.launched = False
        self.last_bearing_sign = 1.0   # which way to search
        # -- Exposed for HUD --
        self.bearing = 0.0
        self.los_rate = 0.0
        self.est_range = 0.0
        self.heading_rate_cmd = 0.0
        self.visible = False

    # -----------------------------------------------------------------

    def act(self, cam_line, aux_vec):
        """Return (left_power, right_power) both in [0, 1].

        cam_line : (CAM_WIDTH,) float32  -- 1D camera observation
        aux_vec  : (3,) float32          -- [gyro_z, accel_fwd, accel_lat]
        """
        p = self.params
        omega = aux_vec[0] * 10.0          # de-normalise gyro -> rad/s

        # -- Detect target in camera --------------------------------
        total = float(cam_line.sum())
        peak = float(cam_line.max()) if total > _MIN_TOTAL else 0.0

        if total > _MIN_TOTAL:
            centre = float((cam_line * _PIXEL_IDX).sum() / total)
            bearing = (centre - _CAM_CENTRE) / CAM_WIDTH * CAM_FOV
            eff_w = total / max(peak, 1e-6)
            ang_size = max(eff_w, 0.5) / CAM_WIDTH * CAM_FOV
            est_range = 2.0 * TARGET_RADIUS / ang_size
            self.visible = True
        else:
            bearing = 0.0
            est_range = 0.0
            self.visible = False

        self.bearing = bearing
        self.est_range = est_range

        # -- Not visible -------------------------------------------
        if not self.visible:
            self.los_rate = 0.0
            self.heading_rate_cmd = 0.0
            self.prev_visible = False
            if not self.launched:
                # Pre-launch scan: hover and rotate to find target
                if p.prescan_thrust > 0:
                    d = p.prescan_yaw
                    L = max(0.0, min(1.0, p.prescan_thrust + d))
                    R = max(0.0, min(1.0, p.prescan_thrust - d))
                    return L, R
                return 0.0, 0.0
            # Post-launch scan: rotate toward last-known side
            d = p.search_yaw * self.last_bearing_sign
            L = max(0.0, min(1.0, p.thrust_min + d))
            R = max(0.0, min(1.0, p.thrust_min - d))
            return L, R

        # -- Launch gate -------------------------------------------
        if not self.launched:
            if peak >= p.launch_min_brightness:
                self.launched = True
                self.prev_bearing = bearing
                self.prev_visible = True
                # First tick: pure vertical boost
                return p.thrust_approach, p.thrust_approach
            return 0.0, 0.0

        # -- LOS rate (inertial) -----------------------------------
        if self.prev_visible:
            bearing_rate = (bearing - self.prev_bearing) / self.dt
        else:
            bearing_rate = 0.0
        los_rate = omega + bearing_rate
        self.los_rate = los_rate

        # -- PN + pursuit heading command --------------------------
        hrc = -p.K_bearing * bearing + (p.N - 1.0) * los_rate
        self.heading_rate_cmd = hrc

        omega_err = hrc - omega
        diff = p.K_rate * omega_err

        # -- Thrust ------------------------------------------------
        bf = max(0.0, math.cos(min(abs(bearing), math.pi / 2)))
        combined = p.thrust_min + (p.thrust_approach - p.thrust_min) * bf

        # IMU-based gravity compensation: when accel_fwd drops below
        # the hover reference (~0.5 normalised), the drone is losing
        # altitude — boost thrust proportionally.
        accel_fwd = aux_vec[1]           # normalised body-frame forward accel
        grav_deficit = 0.5 - accel_fwd   # positive when under-thrusting
        combined += p.K_accel * max(0.0, grav_deficit)
        combined = min(combined, 1.0)

        L = max(0.0, min(1.0, combined + diff))
        R = max(0.0, min(1.0, combined - diff))

        # -- State update ------------------------------------------
        if abs(bearing) > 0.01:
            self.last_bearing_sign = 1.0 if bearing > 0 else -1.0
        self.prev_bearing = bearing
        self.prev_visible = True
        return L, R


# =====================================================================
# Vectorised controller for parallel tuning
# =====================================================================

class BatchPNController:
    """Vectorised PN controller operating on N envs simultaneously."""

    def __init__(self, params_list, envs_per_config, dt=1.0 / 60.0):
        nc = len(params_list)
        self.num_configs = nc
        self.envs_per_config = envs_per_config
        self.total = nc * envs_per_config
        self.dt = dt

        def _expand(attr):
            return np.repeat(
                np.array([getattr(p, attr) for p in params_list]),
                envs_per_config,
            )

        self.N         = _expand("N")
        self.K_bearing = _expand("K_bearing")
        self.K_rate    = _expand("K_rate")
        self.thrust_approach = _expand("thrust_approach")
        self.thrust_min      = _expand("thrust_min")
        self.K_accel         = _expand("K_accel")
        self.launch_thresh   = _expand("launch_min_brightness")
        self.search_yaw      = _expand("search_yaw")
        self.prescan_thrust  = _expand("prescan_thrust")
        self.prescan_yaw     = _expand("prescan_yaw")

        self.reset_all()

    def reset_all(self):
        n = self.total
        self.prev_bearing = np.zeros(n)
        self.prev_visible = np.zeros(n, dtype=bool)
        self.launched      = np.zeros(n, dtype=bool)
        self.last_sign     = np.ones(n)

    def reset_envs(self, mask):
        """Reset controller state for envs indicated by boolean mask."""
        self.prev_bearing[mask] = 0.0
        self.prev_visible[mask] = False
        self.launched[mask]      = False
        self.last_sign[mask]     = 1.0

    # -----------------------------------------------------------------

    def act(self, cam, aux):
        """cam: (N, 64) float32, aux: (N, 3) float32 -> left, right (N,)"""
        omega = aux[:, 0] * 10.0

        # Detection
        total_w = cam.sum(axis=1)                    # (N,)
        peak    = cam.max(axis=1)                    # (N,)
        visible = total_w > _MIN_TOTAL

        safe_total = np.where(visible, total_w, 1.0)
        centre = np.where(visible,
                          (cam * _PIXEL_IDX).sum(axis=1) / safe_total,
                          _CAM_CENTRE)
        bearing = (centre - _CAM_CENTRE) / CAM_WIDTH * CAM_FOV

        # Launch gate
        newly_launch = visible & (peak >= self.launch_thresh) & (~self.launched)
        self.launched |= newly_launch

        grounded = ~self.launched

        # LOS rate
        brg_rate = np.where(
            self.prev_visible & visible,
            (bearing - self.prev_bearing) / self.dt,
            0.0,
        )
        los_rate = omega + brg_rate

        # PN + pursuit
        hrc = -self.K_bearing * bearing + (self.N - 1.0) * los_rate
        omega_err = hrc - omega
        diff = self.K_rate * omega_err

        # Thrust
        bf = np.maximum(0.0, np.cos(np.minimum(np.abs(bearing), math.pi / 2)))
        combined = self.thrust_min + (self.thrust_approach - self.thrust_min) * bf

        # IMU gravity compensation
        accel_fwd = aux[:, 1]
        grav_deficit = np.maximum(0.0, 0.5 - accel_fwd)
        combined = np.minimum(combined + self.K_accel * grav_deficit, 1.0)

        # Search mode (launched but lost)
        searching = self.launched & ~visible
        combined = np.where(searching, self.thrust_min, combined)
        diff     = np.where(searching, self.search_yaw * self.last_sign, diff)

        # Pre-launch: prescan hover+rotate, or zero
        prescanning = grounded & ~visible
        combined = np.where(prescanning, self.prescan_thrust, combined)
        diff     = np.where(prescanning, self.prescan_yaw, diff)
        # Grounded + visible but below launch threshold => still zero
        below_thresh = grounded & visible & (peak < self.launch_thresh)
        combined = np.where(below_thresh, self.prescan_thrust, combined)
        diff     = np.where(below_thresh, self.prescan_yaw, diff)

        left  = np.clip(combined + diff, 0.0, 1.0)
        right = np.clip(combined - diff, 0.0, 1.0)

        # State update
        self.last_sign = np.where(
            visible & (np.abs(bearing) > 0.01),
            np.sign(bearing),
            self.last_sign,
        )
        self.prev_bearing = np.where(visible, bearing, self.prev_bearing)
        self.prev_visible = visible.copy()

        return left, right
