#!/usr/bin/env python3
"""Drone Intercept 2D – Cython-accelerated arcade game.

Pilot your interceptor drone (arrow keys / WASD) and smash into
the enemy drone patrolling high above.
"""

import os
import sys
import math
import array
import random

import numpy as np
import torch
import pygame
import physics  # Cython module
from policy import DronePolicy
from env import CAM_WIDTH as AI_CAM_WIDTH, CAM_FOV as AI_CAM_FOV, NUM_FRAMES, WORLD_H

# ── Display ──────────────────────────────────────────────────────────
WIDTH, HEIGHT = 1024, 768
FPS = 60
GROUND_Y = HEIGHT - 40

# ── Colours ──────────────────────────────────────────────────────────
SKY_TOP = (15, 10, 40)
SKY_BOT = (60, 40, 90)
GROUND = (30, 65, 30)
GROUND_LINE = (50, 100, 50)
WHITE = (255, 255, 255)
HUD_COL = (0, 255, 180)
THRUST_COL = (255, 160, 40)
EXPLOSION_COLS = [(255, 255, 100), (255, 180, 50), (255, 80, 20), (200, 40, 10)]

# ── Drone state indices ──────────────────────────────────────────────
#  [x, y, vx, vy, angle, radius]
PLAYER_RADIUS = 14.0
TARGET_RADIUS = 18.0


def make_state(x, y, radius):
    return array.array("d", [x, y, 0.0, 0.0, 0.0, radius])


# ── Dynamic camera ───────────────────────────────────────────────────

class Camera:
    """Computes a view that keeps both drones on screen with smooth tracking."""

    PADDING = 150       # min pixels of padding around drones on screen
    MIN_ZOOM = 0.15     # don't zoom out more than this
    MAX_ZOOM = 1.0      # 1:1 is the closest
    LERP_SPEED = 3.0    # how fast camera tracks (per second)

    def __init__(self):
        # Camera centre in world coords and current zoom
        self.cx = WIDTH / 2
        self.cy = GROUND_Y / 2
        self.zoom = 1.0

    def update(self, player, target, dt):
        """Smoothly track so both drones are visible."""
        # Desired centre: midpoint of the two drones
        mid_x = (player[0] + target[0]) / 2
        mid_y = (player[1] + target[1]) / 2

        # Required span to fit both drones
        dx = abs(player[0] - target[0]) + self.PADDING * 2
        dy = abs(player[1] - target[1]) + self.PADDING * 2

        zoom_x = WIDTH / max(dx, 1)
        zoom_y = GROUND_Y / max(dy, 1)   # only use sky area for framing
        desired_zoom = min(zoom_x, zoom_y)
        desired_zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, desired_zoom))

        # Smooth interpolation
        t = min(1.0, self.LERP_SPEED * dt)
        self.cx += (mid_x - self.cx) * t
        self.cy += (mid_y - self.cy) * t
        self.zoom += (desired_zoom - self.zoom) * t

    def world_to_screen(self, wx, wy):
        """Convert world coordinates to screen pixel coordinates."""
        sx = (wx - self.cx) * self.zoom + WIDTH / 2
        sy = (wy - self.cy) * self.zoom + GROUND_Y / 2
        return sx, sy

    def scale(self, size):
        """Scale a world-space size to screen pixels."""
        return size * self.zoom


# ── Drawing helpers ──────────────────────────────────────────────────

def draw_gradient_sky(surf):
    """Simple vertical gradient for the sky."""
    for y in range(GROUND_Y):
        t = y / GROUND_Y
        r = int(SKY_TOP[0] + (SKY_BOT[0] - SKY_TOP[0]) * t)
        g = int(SKY_TOP[1] + (SKY_BOT[1] - SKY_TOP[1]) * t)
        b = int(SKY_TOP[2] + (SKY_BOT[2] - SKY_TOP[2]) * t)
        pygame.draw.line(surf, (r, g, b), (0, y), (WIDTH, y))


def _rotated_poly(points, angle, cx, cy):
    """Rotate a list of (x, y) offsets around (cx, cy)."""
    sa, ca = math.sin(angle), math.cos(angle)
    return [(cx + x * ca - y * sa, cy + x * sa + y * ca) for x, y in points]


def draw_drone(surf, state, colour, cam, thrust_on=False):
    """Draw a drone as a stylised quad-copter shape."""
    x, y, vx, vy, angle, radius = state
    sx, sy = cam.world_to_screen(x, y)
    r = cam.scale(radius)

    # Body
    body = [(-r * 0.35, -r * 0.6), (r * 0.35, -r * 0.6),
            (r * 0.4, r * 0.3), (-r * 0.4, r * 0.3)]
    pts = _rotated_poly(body, angle, sx, sy)
    pygame.draw.polygon(surf, colour, pts)
    pygame.draw.polygon(surf, WHITE, pts, 1)

    # Arms + rotors
    rotor_radius = max(1, int(r * 0.35))
    for side in (-1, 1):
        arm_end_x = side * r * 0.9
        arm_end_y = -r * 0.15
        arm = [(0, -r * 0.15), (arm_end_x, arm_end_y)]
        arm_pts = _rotated_poly(arm, angle, sx, sy)
        pygame.draw.line(surf, WHITE, arm_pts[0], arm_pts[1], max(1, int(2 * cam.zoom)))
        # Rotor disc — filled bright when thrusting (spinning), thin outline when idle
        rotor_cx, rotor_cy = _rotated_poly([(arm_end_x, arm_end_y - r * 0.15)], angle, sx, sy)[0]
        rcx, rcy = int(rotor_cx), int(rotor_cy)
        if thrust_on:
            # Motion-blurred spin: filled disc with a faint outer halo
            halo_r = rotor_radius + max(1, int(r * 0.1))
            halo = pygame.Surface((halo_r * 2 + 2, halo_r * 2 + 2), pygame.SRCALPHA)
            pygame.draw.circle(halo, (*THRUST_COL, 70), (halo_r + 1, halo_r + 1), halo_r)
            surf.blit(halo, (rcx - halo_r - 1, rcy - halo_r - 1))
            pygame.draw.circle(surf, THRUST_COL, (rcx, rcy), rotor_radius)
            pygame.draw.circle(surf, WHITE, (rcx, rcy), rotor_radius, 1)
        else:
            pygame.draw.circle(surf, colour, (rcx, rcy), rotor_radius, 1)


def draw_stars(surf, stars):
    for sx, sy, brightness in stars:
        c = int(brightness)
        surf.set_at((sx, sy), (c, c, c))


def draw_ground(surf, cam):
    """Draw the ground plane, accounting for camera."""
    _, ground_sy = cam.world_to_screen(0, GROUND_Y)
    ground_sy = int(ground_sy)
    if ground_sy < HEIGHT:
        pygame.draw.rect(surf, GROUND, (0, ground_sy, WIDTH, HEIGHT - ground_sy))
        # Ground hash lines
        # Figure out world X range visible on screen
        left_wx = cam.cx - (WIDTH / 2) / cam.zoom
        right_wx = cam.cx + (WIDTH / 2) / cam.zoom
        spacing = 60
        start_gx = int(left_wx // spacing) * spacing
        for gwx in range(start_gx, int(right_wx) + spacing, spacing):
            sx1, _ = cam.world_to_screen(gwx, GROUND_Y)
            sx2, sy2 = cam.world_to_screen(gwx + 30, GROUND_Y + 40)
            pygame.draw.line(surf, GROUND_LINE, (int(sx1), ground_sy), (int(sx2), int(sy2)), 1)


def draw_altimeter(surf, font, player_y):
    alt = max(0, int((GROUND_Y - player_y) / 3))
    txt = font.render(f"ALT {alt}m", True, HUD_COL)
    surf.blit(txt, (15, 15))


def draw_speed(surf, font, state):
    spd = int(math.hypot(state[2], state[3]) / 5)
    txt = font.render(f"SPD {spd} km/h", True, HUD_COL)
    surf.blit(txt, (15, 40))


def draw_distance_indicator(surf, font, player, target):
    dist = physics.distance(player, target)
    dist_m = int(dist / 3)
    txt = font.render(f"TGT {dist_m}m", True, (255, 80, 80))
    surf.blit(txt, (15, 65))

    # Direction arrow on HUD
    dx = target[0] - player[0]
    dy = target[1] - player[1]
    a = math.atan2(dy, dx)
    cx, cy = WIDTH - 60, 60
    ar = 25
    ex, ey = cx + math.cos(a) * ar, cy + math.sin(a) * ar
    pygame.draw.circle(surf, (255, 80, 80), (cx, cy), ar + 5, 1)
    pygame.draw.line(surf, (255, 80, 80), (cx, cy), (int(ex), int(ey)), 2)
    pygame.draw.circle(surf, (255, 80, 80), (int(ex), int(ey)), 3)


# ── Camera view ──────────────────────────────────────────────────────
CAM_WIDTH = 300      # pixels wide
CAM_HEIGHT = 40      # pixels tall
CAM_FOV = math.pi / 2  # 90-degree field of view
CAM_BG = (5, 5, 15)
CAM_BORDER = (0, 255, 180)
CAM_TARGET_COL = (255, 60, 60)
CAM_GROUND_COL = (30, 65, 30)


def draw_camera_view(surf, font, player, target):
    """Draw the interceptor's 1D forward-facing camera as a horizontal strip."""
    cam_x = (WIDTH - CAM_WIDTH) // 2
    cam_y = HEIGHT - 100

    # Background
    cam_rect = pygame.Rect(cam_x - 2, cam_y - 2, CAM_WIDTH + 4, CAM_HEIGHT + 4)
    pygame.draw.rect(surf, CAM_BG, (cam_x, cam_y, CAM_WIDTH, CAM_HEIGHT))

    # -- Render ground line in camera view --
    # The ground is at y = GROUND_Y. Compute bearing to ground directly below/ahead.
    # For each column of the camera, cast a ray and check if it hits ground.
    half_fov = CAM_FOV / 2.0
    heading = math.atan2(-math.cos(player[4]), math.sin(player[4]))
    py = player[1]

    # Simple ground rendering: for the bottom portion of the camera,
    # draw green where rays point below the horizon
    for col in range(0, CAM_WIDTH, 2):  # step by 2 for performance
        # Ray angle for this column
        bearing_frac = (col / CAM_WIDTH - 0.5) * 2.0  # -1 to +1
        ray_angle = heading + bearing_frac * half_fov
        # Ray direction: (cos(ray_angle), sin(ray_angle)) in screen coords
        # sin(ray_angle) > 0 means pointing downward (positive Y = down)
        ray_dy = math.sin(ray_angle)
        if ray_dy > 0.05:  # pointing somewhat downward
            # How far to ground
            ground_dist = (GROUND_Y - py) / ray_dy
            if ground_dist > 0:
                # Brightness based on distance
                brightness = max(0.2, min(1.0, 80.0 / ground_dist))
                gc = (int(30 * brightness), int(65 * brightness), int(30 * brightness))
                bar_h = min(CAM_HEIGHT, max(2, int(CAM_HEIGHT * 0.5 * brightness)))
                pygame.draw.rect(surf, gc,
                                 (cam_x + col, cam_y + CAM_HEIGHT - bar_h, 2, bar_h))

    # -- Project target into camera --
    cam_out = array.array("d", [0.0, 0.0, 0.0])
    physics.camera_project(player, target, CAM_FOV, CAM_WIDTH, cam_out)

    centre_px = cam_out[0]
    width_px = cam_out[1]
    bearing = cam_out[2]

    if centre_px >= 0:
        # Target is in view — draw a red bar
        half_w = width_px / 2.0
        left = max(0, int(centre_px - half_w))
        right = min(CAM_WIDTH, int(centre_px + half_w))
        bar_width = max(1, right - left)

        # Brighter when closer
        dist = physics.distance(player, target)
        intensity = max(0.3, min(1.0, 200.0 / max(dist, 1.0)))
        rc = (int(255 * intensity), int(60 * intensity), int(60 * intensity))

        # Draw the red target bar (full height of camera)
        pygame.draw.rect(surf, rc,
                         (cam_x + left, cam_y + 2, bar_width, CAM_HEIGHT - 4))
        # Bright core
        core_left = max(0, int(centre_px - half_w * 0.3))
        core_right = min(CAM_WIDTH, int(centre_px + half_w * 0.3))
        if core_right > core_left:
            pygame.draw.rect(surf, (255, int(120 * intensity), int(120 * intensity)),
                             (cam_x + core_left, cam_y + 4, core_right - core_left, CAM_HEIGHT - 8))

    # Border
    pygame.draw.rect(surf, CAM_BORDER, cam_rect, 1)

    # Centre crosshair tick
    mid_x = cam_x + CAM_WIDTH // 2
    pygame.draw.line(surf, CAM_BORDER, (mid_x, cam_y - 2), (mid_x, cam_y + 4), 1)
    pygame.draw.line(surf, CAM_BORDER, (mid_x, cam_y + CAM_HEIGHT - 4),
                     (mid_x, cam_y + CAM_HEIGHT + 2), 1)

    # Label
    label = font.render("CAMERA", True, CAM_BORDER)
    surf.blit(label, (cam_x, cam_y - 18))

    # Bearing readout
    if centre_px >= 0:
        deg = math.degrees(bearing)
        btxt = font.render(f"{deg:+.1f}\u00b0", True, CAM_TARGET_COL)
    else:
        btxt = font.render("---", True, (100, 100, 100))
    surf.blit(btxt, (cam_x + CAM_WIDTH + 8, cam_y + 10))


# ── Model I/O panel ──────────────────────────────────────────────────
PANEL_W = 230
PANEL_H = 360
PANEL_BG = (5, 5, 20, 210)
PANEL_LABEL = (150, 200, 200)
TURN_COL = (255, 200, 120)
TURN_DIM = (130, 100, 50)
THRUST_COL_BRIGHT = (255, 160, 40)
THRUST_DIM = (130, 80, 20)
VALUE_COL = (150, 255, 150)


def _draw_bipolar_bar(surf, x, y, w, h, value):
    """Draw a centered bar that extends left or right from the midpoint."""
    bg = pygame.Rect(x, y, w, h)
    pygame.draw.rect(surf, (30, 30, 40), bg)
    cx = x + w // 2
    pygame.draw.line(surf, (90, 90, 110), (cx, y - 1), (cx, y + h + 1), 1)
    v = max(-1.0, min(1.0, value))
    bar_len = int(abs(v) * w / 2)
    if v >= 0:
        pygame.draw.rect(surf, HUD_COL, (cx, y, bar_len, h))
    else:
        pygame.draw.rect(surf, HUD_COL, (cx - bar_len, y, bar_len, h))


def draw_model_panel(surf, font, small_font, autopilot):
    """Live visualization of the policy's inputs and outputs."""
    panel_x = WIDTH - PANEL_W - 10
    panel_y = 100

    bg = pygame.Surface((PANEL_W, PANEL_H), pygame.SRCALPHA)
    bg.fill(PANEL_BG)
    surf.blit(bg, (panel_x, panel_y))
    pygame.draw.rect(surf, HUD_COL, (panel_x, panel_y, PANEL_W, PANEL_H), 1)

    title = font.render("MODEL I/O", True, HUD_COL)
    surf.blit(title, (panel_x + 10, panel_y + 6))

    y = panel_y + 30

    # ── Camera frame stack ───────────────────────────────────────────
    label = small_font.render("CAMERA  (4 frames x 64 px)", True, PANEL_LABEL)
    surf.blit(label, (panel_x + 10, y))
    y += 14

    frames = autopilot.frames  # (4, 64)
    cell_w = 3
    cell_h = 9
    frame_gap = 2
    origin_x = panel_x + 10
    bg_rect = pygame.Rect(origin_x - 1, y - 1,
                          cell_w * 64 + 2, 4 * cell_h + 3 * frame_gap + 2)
    pygame.draw.rect(surf, (12, 12, 22), bg_rect)
    # Row 0 is oldest, row 3 is newest
    for i in range(4):
        row_y = y + i * (cell_h + frame_gap)
        for j in range(64):
            v = float(frames[i, j])
            if v > 0.02:
                ic = max(40, min(255, int(v * 255)))
                col = (ic, int(ic * 0.35), int(ic * 0.35))
            else:
                col = (22, 22, 32)
            pygame.draw.rect(surf, col, (origin_x + j * cell_w, row_y, cell_w, cell_h))
    # "newest" tick
    tick_y = y + 3 * (cell_h + frame_gap) + cell_h + 1
    pygame.draw.polygon(surf, HUD_COL,
                        [(origin_x - 4, tick_y - 3), (origin_x - 4, tick_y + 3),
                         (origin_x - 1, tick_y)])
    y += 4 * cell_h + 3 * frame_gap + 8

    # ── IMU ──────────────────────────────────────────────────────────
    label = small_font.render("IMU  (gyro / accel_fwd / accel_lat)", True, PANEL_LABEL)
    surf.blit(label, (panel_x + 10, y))
    y += 14

    aux = autopilot.last_aux_vec
    imu_labels = ("gyro", "a_fwd", "a_lat")
    bar_w = 130
    bar_h = 7
    for lbl, val in zip(imu_labels, aux):
        lt = small_font.render(lbl, True, (180, 180, 180))
        surf.blit(lt, (panel_x + 10, y - 1))
        _draw_bipolar_bar(surf, panel_x + 55, y + 2, bar_w, bar_h, float(val))
        vt = small_font.render(f"{float(val):+.2f}", True, (200, 200, 200))
        surf.blit(vt, (panel_x + 55 + bar_w + 6, y - 1))
        y += 14

    y += 4
    pygame.draw.line(surf, (50, 80, 80),
                     (panel_x + 10, y), (panel_x + PANEL_W - 10, y), 1)
    y += 6

    # ── TURN probabilities ───────────────────────────────────────────
    label = small_font.render("TURN  (left / none / right)", True, TURN_COL)
    surf.blit(label, (panel_x + 10, y))
    y += 14

    turn_probs = autopilot.last_turn_probs
    turn_labels = ("L", "N", "R")
    max_bar = 150
    for i, (tl, p) in enumerate(zip(turn_labels, turn_probs)):
        lt = small_font.render(tl, True, (200, 200, 200))
        surf.blit(lt, (panel_x + 12, y - 1))
        bx = panel_x + 30
        pygame.draw.rect(surf, (30, 30, 40), (bx, y + 2, max_bar, bar_h))
        col = TURN_COL if i == autopilot.last_turn else TURN_DIM
        pygame.draw.rect(surf, col, (bx, y + 2, int(float(p) * max_bar), bar_h))
        pt = small_font.render(f"{int(float(p) * 100):3d}%", True, (200, 200, 200))
        surf.blit(pt, (bx + max_bar + 6, y - 1))
        y += 13

    y += 4

    # ── THRUST probabilities ─────────────────────────────────────────
    label = small_font.render("THRUST  (off / on)", True, THRUST_COL_BRIGHT)
    surf.blit(label, (panel_x + 10, y))
    y += 14

    thrust_probs = autopilot.last_thrust_probs
    thrust_labels = ("OFF", "ON")
    for i, (tl, p) in enumerate(zip(thrust_labels, thrust_probs)):
        lt = small_font.render(tl, True, (200, 200, 200))
        surf.blit(lt, (panel_x + 12, y - 1))
        bx = panel_x + 42
        pygame.draw.rect(surf, (30, 30, 40), (bx, y + 2, max_bar - 12, bar_h))
        col = THRUST_COL_BRIGHT if i == autopilot.last_thrust else THRUST_DIM
        pygame.draw.rect(surf, col, (bx, y + 2, int(float(p) * (max_bar - 12)), bar_h))
        pt = small_font.render(f"{int(float(p) * 100):3d}%", True, (200, 200, 200))
        surf.blit(pt, (bx + (max_bar - 12) + 6, y - 1))
        y += 13

    y += 4

    # ── Value estimate ───────────────────────────────────────────────
    v_label = small_font.render("VALUE", True, VALUE_COL)
    surf.blit(v_label, (panel_x + 10, y))
    v_txt = font.render(f"{autopilot.last_value:+.2f}", True, VALUE_COL)
    surf.blit(v_txt, (panel_x + 60, y - 3))


# ── Particle explosion ──────────────────────────────────────────────

class Particle:
    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "size")

    def __init__(self, x, y):
        a = random.uniform(0, 2 * math.pi)
        spd = random.uniform(50, 350)
        self.x = x
        self.y = y
        self.vx = math.cos(a) * spd
        self.vy = math.sin(a) * spd
        self.life = random.uniform(0.4, 1.2)
        self.max_life = self.life
        self.size = random.randint(2, 6)

    def update(self, dt):
        self.vy += 120 * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.life -= dt

    def draw(self, surf, cam):
        t = self.life / self.max_life
        idx = min(int((1 - t) * len(EXPLOSION_COLS)), len(EXPLOSION_COLS) - 1)
        col = EXPLOSION_COLS[idx]
        sz = max(1, int(cam.scale(self.size) * t))
        sx, sy = cam.world_to_screen(self.x, self.y)
        pygame.draw.circle(surf, col, (int(sx), int(sy)), sz)


# ── AI autopilot ─────────────────────────────────────────────────────

CHECKPOINT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "checkpoints", "policy_final.pt")


class Autopilot:
    """Wraps the trained policy for use inside the game loop."""

    GRAVITY = 300.0  # must match physics.pyx

    def __init__(self):
        self.policy = None
        self.frames = np.zeros((NUM_FRAMES, AI_CAM_WIDTH), dtype=np.float32)
        self.cam_out = array.array("d", [0.0, 0.0, 0.0])
        self.last_turn = 1   # 0=left 1=none 2=right
        self.last_thrust = 0
        # IMU state tracking
        self.prev_vx = 0.0
        self.prev_vy = 0.0
        self.prev_angle = 0.0
        # Model I/O snapshots for visualization
        self.last_aux_vec = np.zeros(3, dtype=np.float32)
        self.last_turn_probs = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.last_thrust_probs = np.array([1.0, 0.0], dtype=np.float32)
        self.last_value = 0.0

    def load(self):
        if self.policy is not None:
            return True
        if not os.path.exists(CHECKPOINT_PATH):
            return False
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        self.policy = DronePolicy()
        self.policy.load_state_dict(ckpt["policy"])
        self.policy.eval()
        return True

    def reset(self):
        self.frames[:] = 0
        self.last_turn = 1
        self.last_thrust = 0
        self.prev_vx = 0.0
        self.prev_vy = 0.0
        self.prev_angle = 0.0
        self.last_aux_vec[:] = 0.0
        self.last_turn_probs[:] = [0.0, 1.0, 0.0]
        self.last_thrust_probs[:] = [1.0, 0.0]
        self.last_value = 0.0

    def _render_camera_line(self, player, target):
        line = np.zeros(AI_CAM_WIDTH, dtype=np.float32)
        physics.camera_project(player, target, AI_CAM_FOV, AI_CAM_WIDTH, self.cam_out)
        centre = self.cam_out[0]
        width = self.cam_out[1]
        if centre >= 0:
            dist = physics.distance(player, target)
            intensity = min(1.0, max(0.2, 200.0 / max(dist, 1.0)))
            half_w = width / 2.0
            left = max(0, int(centre - half_w))
            right = min(AI_CAM_WIDTH, int(centre + half_w) + 1)
            line[left:right] = intensity
        return line

    def _get_aux(self, player):
        """IMU sensor: gyro_z, accel_forward, accel_lateral in body frame."""
        dt = 1.0 / 60.0
        vx, vy = player[2], player[3]
        angle = player[4]

        gyro_z = (angle - self.prev_angle) / dt

        ax_world = (vx - self.prev_vx) / dt
        ay_world = (vy - self.prev_vy) / dt
        ax_proper = ax_world
        ay_proper = ay_world - self.GRAVITY

        sa = math.sin(angle)
        ca = math.cos(angle)
        accel_fwd = ax_proper * sa - ay_proper * ca
        accel_lat = ax_proper * ca + ay_proper * sa

        self.prev_vx = vx
        self.prev_vy = vy
        self.prev_angle = angle

        return np.array([gyro_z / 10.0, accel_fwd / 600.0, accel_lat / 600.0],
                        dtype=np.float32)

    @torch.no_grad()
    def act(self, player, target):
        """Return (turn, thrust) where turn is -1/0/+1 and thrust is 0/1."""
        new_line = self._render_camera_line(player, target)
        self.frames[:-1] = self.frames[1:]
        self.frames[-1] = new_line

        aux = self._get_aux(player)
        self.last_aux_vec = aux.copy()

        cam_t = torch.from_numpy(self.frames).unsqueeze(0)
        aux_t = torch.from_numpy(aux).unsqueeze(0)

        # Forward pass directly so we can extract logits + value for the HUD
        turn_logits, thrust_logits, value = self.policy(cam_t, aux_t)
        self.last_turn_probs = torch.softmax(turn_logits, dim=-1).squeeze(0).numpy()
        self.last_thrust_probs = torch.softmax(thrust_logits, dim=-1).squeeze(0).numpy()
        self.last_value = float(value.item())

        # Sample actions (matches training-time behavior)
        a_turn = torch.distributions.Categorical(logits=turn_logits).sample()
        a_thrust = torch.distributions.Categorical(logits=thrust_logits).sample()

        self.last_turn = a_turn.item()
        self.last_thrust = a_thrust.item()
        turn = self.last_turn - 1   # 0,1,2 → -1,0,+1
        thrust = self.last_thrust
        return turn, thrust


# ── Game states ──────────────────────────────────────────────────────
STATE_TITLE = 0
STATE_PLAY = 1
STATE_HIT = 2   # explosion / score screen


def run():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Drone Intercept 2D")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("menlo", 18)
    small_font = pygame.font.SysFont("menlo", 12)
    big_font = pygame.font.SysFont("menlo", 36, bold=True)

    # Pre-render sky
    sky_surf = pygame.Surface((WIDTH, GROUND_Y))
    draw_gradient_sky(sky_surf)

    # Stars
    stars = [(random.randint(0, WIDTH), random.randint(0, GROUND_Y - 50),
              random.randint(80, 255)) for _ in range(120)]

    # ── AI autopilot ─────────────────────────────────────────────────
    autopilot = Autopilot()
    auto_mode = False

    # ── Dynamic camera ───────────────────────────────────────────────
    cam = Camera()

    # ── State init ───────────────────────────────────────────────────
    game_state = STATE_TITLE
    player = make_state(WIDTH / 2, GROUND_Y - 30, PLAYER_RADIUS)
    target = make_state(WIDTH / 2, 150, TARGET_RADIUS)
    target_phase = 0.0
    target_alt = 160.0
    target_speed = 100.0
    patrol_left = 100.0
    patrol_right = WIDTH - 100.0
    particles = []
    score = 0
    attempts = 0
    hit_timer = 0.0

    def reset_round():
        nonlocal target_phase, target_alt, target_speed, patrol_left, patrol_right
        player[0] = WIDTH / 2
        player[1] = GROUND_Y - 30
        player[2] = player[3] = player[4] = 0.0
        target_phase = random.uniform(0, 2 * math.pi)
        target_alt = random.uniform(80, 250)
        target_speed = random.uniform(60, 150)
        centre_x = random.uniform(300, WIDTH - 300)
        patrol_half = random.uniform(200, 400)
        patrol_left = centre_x - patrol_half
        patrol_right = centre_x + patrol_half
        # Start target along the patrol route
        target[0] = random.uniform(patrol_left, patrol_right)
        target[1] = target_alt
        target[2] = target_speed * random.choice([-1, 1])
        target[3] = 0.0
        target[4] = 0.0

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        dt = min(dt, 0.05)  # prevent spiral-of-death

        # ── Events ───────────────────────────────────────────────────
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                if ev.key == pygame.K_c:
                    if auto_mode:
                        auto_mode = False
                    elif autopilot.load():
                        auto_mode = True
                        autopilot.reset()
                if game_state == STATE_TITLE and ev.key in (pygame.K_SPACE, pygame.K_RETURN):
                    game_state = STATE_PLAY
                    reset_round()
                    if auto_mode:
                        autopilot.reset()
                if game_state == STATE_HIT and ev.key in (pygame.K_SPACE, pygame.K_RETURN):
                    game_state = STATE_PLAY
                    reset_round()
                    if auto_mode:
                        autopilot.reset()

        keys = pygame.key.get_pressed()

        # ── Update ───────────────────────────────────────────────────
        if game_state == STATE_PLAY:
            if auto_mode:
                turn, thrust = autopilot.act(player, target)
            else:
                turn = 0
                thrust = 0
                if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                    turn = -1
                if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                    turn = 1
                if keys[pygame.K_UP] or keys[pygame.K_w]:
                    thrust = 1

            physics.player_update(player, dt, turn, thrust)
            physics.clamp_world(player, float(GROUND_Y))

            target_phase += dt
            physics.target_update(target, dt, target_alt, target_speed,
                                  patrol_left, patrol_right, target_phase)

            if physics.check_collision(player, target):
                rel_spd = physics.relative_speed(player, target)
                hit_score = int(rel_spd / 2)
                score += hit_score
                attempts += 1
                # Spawn explosion
                cx = (player[0] + target[0]) / 2
                cy = (player[1] + target[1]) / 2
                particles = [Particle(cx, cy) for _ in range(80)]
                game_state = STATE_HIT
                hit_timer = 0.0

        elif game_state == STATE_HIT:
            hit_timer += dt
            for p in particles:
                p.update(dt)
            particles = [p for p in particles if p.life > 0]

        # ── Update camera ────────────────────────────────────────────
        if game_state in (STATE_PLAY, STATE_HIT):
            cam.update(player, target, dt)

        # ── Draw ─────────────────────────────────────────────────────
        screen.blit(sky_surf, (0, 0))
        draw_stars(screen, stars)
        draw_ground(screen, cam)

        if game_state == STATE_TITLE:
            title = big_font.render("DRONE INTERCEPT", True, WHITE)
            sub = font.render("Arrow keys / WASD to fly.  [C] toggle AI pilot.", True, HUD_COL)
            start = font.render("Press SPACE to launch", True, (255, 255, 100))
            screen.blit(title, (WIDTH // 2 - title.get_width() // 2, HEIGHT // 2 - 60))
            screen.blit(sub, (WIDTH // 2 - sub.get_width() // 2, HEIGHT // 2))
            screen.blit(start, (WIDTH // 2 - start.get_width() // 2, HEIGHT // 2 + 40))
            # Draw demo drones (screen coords, use a static camera)
            title_cam = Camera()
            draw_drone(screen, make_state(WIDTH // 2 - 100, HEIGHT // 2 - 120, PLAYER_RADIUS),
                       (0, 180, 255), title_cam)
            draw_drone(screen, make_state(WIDTH // 2 + 100, HEIGHT // 2 - 120, TARGET_RADIUS),
                       (255, 60, 60), title_cam)

        elif game_state == STATE_PLAY:
            if auto_mode:
                thrust_on = (thrust == 1)
            else:
                thrust_on = keys[pygame.K_UP] or keys[pygame.K_w]
            draw_drone(screen, target, (255, 60, 60), cam)
            draw_drone(screen, player, (0, 180, 255), cam, thrust_on)
            draw_altimeter(screen, font, player[1])
            draw_speed(screen, font, player)
            draw_distance_indicator(screen, font, player, target)
            draw_camera_view(screen, font, player, target)
            score_txt = font.render(f"SCORE {score}", True, HUD_COL)
            screen.blit(score_txt, (WIDTH - score_txt.get_width() - 15, 15))

            # Zoom indicator
            zoom_txt = font.render(f"ZOOM {cam.zoom:.2f}x", True, (100, 100, 100))
            screen.blit(zoom_txt, (15, 90))

            if auto_mode:
                # AI pilot label
                ai_label = font.render("AI PILOT  [C] manual", True, (255, 255, 100))
                screen.blit(ai_label, (WIDTH // 2 - ai_label.get_width() // 2, 15))
                # Show current action
                actions = []
                if turn == -1:
                    actions.append("LEFT")
                elif turn == 1:
                    actions.append("RIGHT")
                if thrust == 1:
                    actions.append("THRUST")
                act_str = " + ".join(actions) if actions else "COAST"
                act_txt = font.render(act_str, True, WHITE)
                screen.blit(act_txt, (WIDTH // 2 - act_txt.get_width() // 2, 38))
                draw_model_panel(screen, font, small_font, autopilot)
            else:
                mode_txt = font.render("[C] auto pilot", True, (100, 100, 100))
                screen.blit(mode_txt, (WIDTH // 2 - mode_txt.get_width() // 2, 15))

        elif game_state == STATE_HIT:
            # Draw drones fading / exploding
            for p in particles:
                p.draw(screen, cam)
            hit_txt = big_font.render(f"HIT!  +{int(physics.relative_speed(player, target) / 2)}",
                                      True, (255, 255, 100))
            screen.blit(hit_txt, (WIDTH // 2 - hit_txt.get_width() // 2, HEIGHT // 2 - 30))
            score_txt = font.render(f"Total score: {score}  |  Hits: {attempts}", True, HUD_COL)
            screen.blit(score_txt, (WIDTH // 2 - score_txt.get_width() // 2, HEIGHT // 2 + 20))
            if hit_timer > 1.0:
                cont = font.render("Press SPACE to go again", True, WHITE)
                screen.blit(cont, (WIDTH // 2 - cont.get_width() // 2, HEIGHT // 2 + 60))

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    run()
