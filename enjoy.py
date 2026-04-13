#!/usr/bin/env python3
"""Watch the PN guidance controller play the game with the full pygame renderer."""

import sys
import math
import array
import argparse

import numpy as np
import pygame

import physics
from env import DroneInterceptEnv, CAM_WIDTH, WORLD_W, WORLD_H
from pn_controller import PNController, load_params
from game import (WIDTH, HEIGHT, GROUND_Y, FPS, WHITE, HUD_COL,
                  draw_gradient_sky, draw_stars, draw_ground, draw_drone,
                  draw_altimeter, draw_speed, draw_distance_indicator,
                  draw_camera_view, make_state, PLAYER_RADIUS, TARGET_RADIUS,
                  Camera)
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="pn_params.json",
                        help="Path to PN parameters JSON")
    args = parser.parse_args()

    params = load_params(args.params)
    controller = PNController(params)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Drone Intercept 2D – PN Guidance")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("menlo", 18)

    sky_surf = pygame.Surface((WIDTH, GROUND_Y))
    draw_gradient_sky(sky_surf)
    stars = [(random.randint(0, WIDTH), random.randint(0, GROUND_Y - 50),
              random.randint(80, 255)) for _ in range(120)]

    cam = Camera()
    env = DroneInterceptEnv(seed=0)
    obs = env.reset()
    score = 0
    hits = 0

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        dt = min(dt, 0.05)

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                running = False
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_r:
                obs = env.reset()
                controller.reset()
                score = 0
                hits = 0

        # PN controller inference
        aux = env.get_aux()
        left_power, right_power = controller.act(obs, aux)

        obs, reward, done, truncated = env.step(left_power, right_power)
        score += reward

        if done or truncated:
            if done:
                hits += 1
            obs = env.reset()
            controller.reset()

        p = env.player
        t = env.target

        cam.update(p, t, dt)

        # ── Render ───────────────────────────────────────────────────
        screen.blit(sky_surf, (0, 0))
        draw_stars(screen, stars)
        draw_ground(screen, cam)

        draw_drone(screen, t, (255, 60, 60), cam)
        draw_drone(screen, p, (0, 180, 255), cam, left_power, right_power)
        draw_altimeter(screen, font, p[1])
        draw_speed(screen, font, p)
        draw_distance_indicator(screen, font, p, t)
        draw_camera_view(screen, font, p, t)

        score_txt = font.render(f"SCORE {int(score)}  HITS {hits}", True, HUD_COL)
        screen.blit(score_txt, (WIDTH - score_txt.get_width() - 15, 15))

        ai_label = font.render("PN GUIDANCE", True, (255, 255, 100))
        screen.blit(ai_label, (WIDTH // 2 - ai_label.get_width() // 2, 15))

        status = "TRACK" if controller.visible else ("WAIT" if not controller.launched else "SEARCH")
        info = (f"L {left_power:.2f}  R {right_power:.2f}  "
                f"brg {math.degrees(controller.bearing):+.1f}  "
                f"rng {controller.est_range:.0f}  [{status}]")
        act_txt = font.render(info, True, WHITE)
        screen.blit(act_txt, (WIDTH // 2 - act_txt.get_width() // 2, 40))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
