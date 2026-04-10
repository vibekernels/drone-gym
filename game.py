#!/usr/bin/env python3
"""Drone Intercept 2D – Cython-accelerated arcade game.

Pilot your interceptor drone (arrow keys / WASD) and smash into
the enemy drone patrolling high above.
"""

import sys
import math
import array
import random

import pygame
import physics  # Cython module

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


def draw_drone(surf, state, colour, thrust_on=False):
    """Draw a drone as a stylised quad-copter shape."""
    x, y, vx, vy, angle, radius = state
    r = radius

    # Body
    body = [(-r * 0.35, -r * 0.6), (r * 0.35, -r * 0.6),
            (r * 0.4, r * 0.3), (-r * 0.4, r * 0.3)]
    pts = _rotated_poly(body, angle, x, y)
    pygame.draw.polygon(surf, colour, pts)
    pygame.draw.polygon(surf, WHITE, pts, 1)

    # Arms + rotors
    for side in (-1, 1):
        arm_end_x = side * r * 0.9
        arm_end_y = -r * 0.15
        arm = [(0, -r * 0.15), (arm_end_x, arm_end_y)]
        arm_pts = _rotated_poly(arm, angle, x, y)
        pygame.draw.line(surf, WHITE, arm_pts[0], arm_pts[1], 2)
        # Rotor disc
        rotor_cx, rotor_cy = _rotated_poly([(arm_end_x, arm_end_y - r * 0.15)], angle, x, y)[0]
        pygame.draw.circle(surf, (*colour, 180), (int(rotor_cx), int(rotor_cy)), int(r * 0.35), 1)

    # Thrust flame
    if thrust_on:
        flame_len = random.uniform(r * 0.6, r * 1.2)
        flame = [(- r * 0.15, r * 0.3), (r * 0.15, r * 0.3), (0, r * 0.3 + flame_len)]
        flame_pts = _rotated_poly(flame, angle, x, y)
        pygame.draw.polygon(surf, THRUST_COL, flame_pts)


def draw_stars(surf, stars):
    for sx, sy, brightness in stars:
        c = int(brightness)
        surf.set_at((sx, sy), (c, c, c))


def draw_ground(surf):
    pygame.draw.rect(surf, GROUND, (0, GROUND_Y, WIDTH, HEIGHT - GROUND_Y))
    for gx in range(0, WIDTH, 60):
        pygame.draw.line(surf, GROUND_LINE, (gx, GROUND_Y), (gx + 30, HEIGHT), 1)


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

    def draw(self, surf):
        t = self.life / self.max_life
        idx = min(int((1 - t) * len(EXPLOSION_COLS)), len(EXPLOSION_COLS) - 1)
        col = EXPLOSION_COLS[idx]
        sz = max(1, int(self.size * t))
        pygame.draw.circle(surf, col, (int(self.x), int(self.y)), sz)


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
    big_font = pygame.font.SysFont("menlo", 36, bold=True)

    # Pre-render sky
    sky_surf = pygame.Surface((WIDTH, GROUND_Y))
    draw_gradient_sky(sky_surf)

    # Stars
    stars = [(random.randint(0, WIDTH), random.randint(0, GROUND_Y - 50),
              random.randint(80, 255)) for _ in range(120)]

    # ── State init ───────────────────────────────────────────────────
    game_state = STATE_TITLE
    player = make_state(WIDTH / 2, GROUND_Y - 30, PLAYER_RADIUS)
    target = make_state(WIDTH / 2, 150, TARGET_RADIUS)
    target_phase = 0.0
    orbit_cx, orbit_cy = WIDTH / 2, 160.0
    orbit_radius = 320.0
    orbit_speed = 0.7
    particles = []
    score = 0
    attempts = 0
    hit_timer = 0.0

    def reset_round():
        nonlocal target_phase, orbit_cx, orbit_cy, orbit_speed
        player[0] = WIDTH / 2
        player[1] = GROUND_Y - 30
        player[2] = player[3] = player[4] = 0.0
        target_phase = random.uniform(0, 2 * math.pi)
        orbit_cx = random.uniform(200, WIDTH - 200)
        orbit_cy = random.uniform(80, 250)
        orbit_speed = random.uniform(0.5, 1.1)

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
                if game_state == STATE_TITLE and ev.key in (pygame.K_SPACE, pygame.K_RETURN):
                    game_state = STATE_PLAY
                    reset_round()
                if game_state == STATE_HIT and ev.key in (pygame.K_SPACE, pygame.K_RETURN):
                    game_state = STATE_PLAY
                    reset_round()

        keys = pygame.key.get_pressed()

        # ── Update ───────────────────────────────────────────────────
        if game_state == STATE_PLAY:
            turn = 0
            thrust = 0
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                turn = -1
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                turn = 1
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                thrust = 1

            physics.player_update(player, dt, turn, thrust)
            physics.wrap_world(player, float(WIDTH), float(GROUND_Y))

            target_phase += orbit_speed * dt
            physics.target_update(target, dt, orbit_cx, orbit_cy,
                                  orbit_radius, orbit_speed, target_phase)

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

        # ── Draw ─────────────────────────────────────────────────────
        screen.blit(sky_surf, (0, 0))
        draw_stars(screen, stars)
        draw_ground(screen)

        if game_state == STATE_TITLE:
            title = big_font.render("DRONE INTERCEPT", True, WHITE)
            sub = font.render("Arrow keys / WASD to fly.  Smash the target drone!", True, HUD_COL)
            start = font.render("Press SPACE to launch", True, (255, 255, 100))
            screen.blit(title, (WIDTH // 2 - title.get_width() // 2, HEIGHT // 2 - 60))
            screen.blit(sub, (WIDTH // 2 - sub.get_width() // 2, HEIGHT // 2))
            screen.blit(start, (WIDTH // 2 - start.get_width() // 2, HEIGHT // 2 + 40))
            # Draw demo drones
            draw_drone(screen, make_state(WIDTH // 2 - 100, HEIGHT // 2 - 120, PLAYER_RADIUS),
                       (0, 180, 255))
            draw_drone(screen, make_state(WIDTH // 2 + 100, HEIGHT // 2 - 120, TARGET_RADIUS),
                       (255, 60, 60))

        elif game_state == STATE_PLAY:
            thrust_on = keys[pygame.K_UP] or keys[pygame.K_w]
            draw_drone(screen, target, (255, 60, 60))
            draw_drone(screen, player, (0, 180, 255), thrust_on)
            draw_altimeter(screen, font, player[1])
            draw_speed(screen, font, player)
            draw_distance_indicator(screen, font, player, target)
            draw_camera_view(screen, font, player, target)
            score_txt = font.render(f"SCORE {score}", True, HUD_COL)
            screen.blit(score_txt, (WIDTH - score_txt.get_width() - 15, 15))

        elif game_state == STATE_HIT:
            # Draw drones fading / exploding
            for p in particles:
                p.draw(screen)
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
