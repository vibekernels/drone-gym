"""Headless drone-intercept environment with vectorized batch support.

Observation: stacked 1D camera frames → (num_frames, cam_width) uint8 image
             + small aux vector (angular_vel, altitude, vert_speed, distance)
Action:      MultiDiscrete([3, 2]) → (turn: left/none/right, thrust: off/on)
"""

import math
import array
import random

import numpy as np
import physics  # Cython module


# ── Environment constants ────────────────────────────────────────────
WORLD_W = 1024.0
WORLD_H = 728.0       # ground Y (same as HEIGHT - 40)
CAM_WIDTH = 64         # downsampled camera resolution
CAM_FOV = math.pi / 2  # 90° field of view
NUM_FRAMES = 4         # stacked frames for temporal info
MAX_STEPS = 600        # 10 seconds at 60 fps
PLAYER_RADIUS = 14.0
TARGET_RADIUS = 18.0

# Reward shaping
REWARD_HIT = 10.0
REWARD_STEP = -0.005         # small time penalty
REWARD_BEARING = 0.02        # per-step bonus for pointing at target
REWARD_CLOSING = 0.01        # per-step bonus for closing distance


class DroneInterceptEnv:
    """Single headless environment instance."""

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        # Pre-allocate Cython state arrays
        self.player = array.array("d", [0.0] * 6)
        self.target = array.array("d", [0.0] * 6)
        self.cam_out = array.array("d", [0.0, 0.0, 0.0])
        # Frame stack buffer
        self.frames = np.zeros((NUM_FRAMES, CAM_WIDTH), dtype=np.float32)
        self.step_count = 0
        self.prev_dist = 0.0
        self.target_phase = 0.0
        self.orbit_cx = 0.0
        self.orbit_cy = 0.0
        self.orbit_radius = 0.0
        self.orbit_speed = 0.0

    def reset(self):
        # Player starts on the ground, facing up
        self.player[0] = WORLD_W / 2
        self.player[1] = WORLD_H - 30
        self.player[2] = 0.0
        self.player[3] = 0.0
        self.player[4] = 0.0
        self.player[5] = PLAYER_RADIUS

        # Randomize target orbit
        self.target_phase = self.rng.uniform(0, 2 * math.pi)
        self.orbit_cx = self.rng.uniform(200, WORLD_W - 200)
        self.orbit_cy = self.rng.uniform(80, 250)
        self.orbit_radius = 320.0
        self.orbit_speed = self.rng.uniform(0.5, 1.1)

        self.target[5] = TARGET_RADIUS
        physics.target_update(self.target, 0.0, self.orbit_cx, self.orbit_cy,
                              self.orbit_radius, self.orbit_speed, self.target_phase)

        self.prev_dist = physics.distance(self.player, self.target)
        self.step_count = 0
        self.frames[:] = 0

        obs = self._get_obs()
        return obs

    def step(self, action_turn, action_thrust):
        """action_turn: 0=left, 1=none, 2=right.  action_thrust: 0=off, 1=on."""
        dt = 1.0 / 60.0
        turn = action_turn - 1  # map 0,1,2 → -1,0,+1
        thrust = action_thrust

        physics.player_update(self.player, dt, turn, thrust)
        physics.clamp_world(self.player, WORLD_H)

        self.target_phase += self.orbit_speed * dt
        physics.target_update(self.target, dt, self.orbit_cx, self.orbit_cy,
                              self.orbit_radius, self.orbit_speed, self.target_phase)

        self.step_count += 1

        # ── Reward ───────────────────────────────────────────────────
        reward = REWARD_STEP

        dist = physics.distance(self.player, self.target)

        # Bearing reward: bonus when target is near camera centre
        physics.camera_project(self.player, self.target,
                               CAM_FOV, CAM_WIDTH, self.cam_out)
        if self.cam_out[0] >= 0:
            # Target in view — reward inversely proportional to bearing
            bearing_frac = abs(self.cam_out[2]) / (CAM_FOV / 2)
            reward += REWARD_BEARING * (1.0 - bearing_frac)

        # Closing reward
        dist_delta = self.prev_dist - dist
        reward += REWARD_CLOSING * dist_delta / 10.0
        self.prev_dist = dist

        # Collision check
        done = False
        if physics.check_collision(self.player, self.target):
            reward += REWARD_HIT
            done = True

        truncated = False
        if self.step_count >= MAX_STEPS:
            truncated = True

        obs = self._get_obs()
        return obs, reward, done, truncated

    def _render_camera_line(self):
        """Render a single 1D camera frame as float32 array in [0, 1]."""
        line = np.zeros(CAM_WIDTH, dtype=np.float32)

        physics.camera_project(self.player, self.target,
                               CAM_FOV, CAM_WIDTH, self.cam_out)
        centre = self.cam_out[0]
        width = self.cam_out[1]

        if centre >= 0:
            dist = physics.distance(self.player, self.target)
            intensity = min(1.0, max(0.2, 200.0 / max(dist, 1.0)))
            half_w = width / 2.0
            left = max(0, int(centre - half_w))
            right = min(CAM_WIDTH, int(centre + half_w) + 1)
            line[left:right] = intensity

        return line

    def _get_obs(self):
        """Return stacked camera frames (NUM_FRAMES, CAM_WIDTH)."""
        new_line = self._render_camera_line()
        # Shift stack and insert new frame
        self.frames[:-1] = self.frames[1:]
        self.frames[-1] = new_line
        return self.frames.copy()

    def get_aux(self):
        """Auxiliary observations: (angular_vel_norm, altitude_norm, vspeed_norm, dist_norm)."""
        ang_vel = self.player[4] / math.pi      # roughly [-1, 1]
        alt = 1.0 - self.player[1] / WORLD_H    # 0=ground, 1=top
        vspeed = -self.player[3] / 400.0         # positive=climbing
        dist = physics.distance(self.player, self.target) / 800.0
        return np.array([ang_vel, alt, vspeed, dist], dtype=np.float32)


# ── Vectorized batch environment ─────────────────────────────────────

class VecEnv:
    """Runs N environments in lock-step with pre-allocated numpy buffers.

    Inspired by PufferLib's vectorization but single-process since our
    Cython physics is cheap enough that multiprocessing overhead dominates.
    """

    def __init__(self, num_envs, seed=0):
        self.num_envs = num_envs
        self.envs = [DroneInterceptEnv(seed=seed + i) for i in range(num_envs)]

        # Pre-allocated output buffers
        self.obs_buf = np.zeros((num_envs, NUM_FRAMES, CAM_WIDTH), dtype=np.float32)
        self.aux_buf = np.zeros((num_envs, 4), dtype=np.float32)
        self.reward_buf = np.zeros(num_envs, dtype=np.float32)
        self.done_buf = np.zeros(num_envs, dtype=bool)
        self.trunc_buf = np.zeros(num_envs, dtype=bool)

        # Episode stats tracking
        self.ep_returns = np.zeros(num_envs, dtype=np.float32)
        self.ep_lengths = np.zeros(num_envs, dtype=np.int32)
        self.completed_returns = []
        self.completed_lengths = []

    def reset(self):
        for i, env in enumerate(self.envs):
            self.obs_buf[i] = env.reset()
            self.aux_buf[i] = env.get_aux()
        self.ep_returns[:] = 0
        self.ep_lengths[:] = 0
        self.completed_returns.clear()
        self.completed_lengths.clear()
        return self.obs_buf.copy(), self.aux_buf.copy()

    def step(self, actions_turn, actions_thrust):
        """Step all environments. Auto-resets on done/truncated.

        actions_turn:   (num_envs,) int array, values in {0, 1, 2}
        actions_thrust: (num_envs,) int array, values in {0, 1}
        """
        for i, env in enumerate(self.envs):
            obs, reward, done, truncated = env.step(
                int(actions_turn[i]), int(actions_thrust[i])
            )
            self.obs_buf[i] = obs
            self.reward_buf[i] = reward
            self.done_buf[i] = done
            self.trunc_buf[i] = truncated

            self.ep_returns[i] += reward
            self.ep_lengths[i] += 1

            if done or truncated:
                self.completed_returns.append(float(self.ep_returns[i]))
                self.completed_lengths.append(int(self.ep_lengths[i]))
                self.ep_returns[i] = 0
                self.ep_lengths[i] = 0
                self.obs_buf[i] = env.reset()

            self.aux_buf[i] = env.get_aux()

        return (self.obs_buf.copy(), self.aux_buf.copy(),
                self.reward_buf.copy(), self.done_buf.copy(), self.trunc_buf.copy())

    def pop_completed_episodes(self):
        """Return and clear completed episode stats."""
        ret = list(self.completed_returns)
        lens = list(self.completed_lengths)
        self.completed_returns.clear()
        self.completed_lengths.clear()
        return ret, lens
