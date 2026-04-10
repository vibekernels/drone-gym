#!/usr/bin/env python3
"""Watch a trained policy play the game with the full pygame renderer."""

import sys
import math
import array
import argparse

import numpy as np
import torch
import pygame

import physics
from env import DroneInterceptEnv, NUM_FRAMES, CAM_WIDTH, WORLD_W, WORLD_H
from policy import DronePolicy
from game import (WIDTH, HEIGHT, GROUND_Y, FPS, WHITE, HUD_COL,
                  draw_gradient_sky, draw_stars, draw_ground, draw_drone,
                  draw_altimeter, draw_speed, draw_distance_indicator,
                  draw_camera_view, make_state, PLAYER_RADIUS, TARGET_RADIUS,
                  Camera)
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to .pt checkpoint")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    policy = DronePolicy().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Drone Intercept 2D – AI Agent")
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
                score = 0
                hits = 0

        # Policy inference
        cam_t = torch.from_numpy(obs).unsqueeze(0).to(device)
        aux_t = torch.from_numpy(env.get_aux()).unsqueeze(0).to(device)
        with torch.no_grad():
            a_turn, a_thrust, _, _, _ = policy.get_action_and_value(cam_t, aux_t)

        obs, reward, done, truncated = env.step(a_turn.item(), a_thrust.item())
        score += reward

        if done:
            hits += 1
            obs = env.reset()
        elif truncated:
            obs = env.reset()

        p = env.player
        t = env.target
        thrust_on = (a_thrust.item() == 1)

        cam.update(p, t, dt)

        # ── Render ───────────────────────────────────────────────────
        screen.blit(sky_surf, (0, 0))
        draw_stars(screen, stars)
        draw_ground(screen, cam)

        draw_drone(screen, t, (255, 60, 60), cam)
        draw_drone(screen, p, (0, 180, 255), cam, thrust_on)
        draw_altimeter(screen, font, p[1])
        draw_speed(screen, font, p)
        draw_distance_indicator(screen, font, p, t)
        draw_camera_view(screen, font, p, t)

        score_txt = font.render(f"SCORE {int(score)}  HITS {hits}", True, HUD_COL)
        screen.blit(score_txt, (WIDTH - score_txt.get_width() - 15, 15))

        ai_label = font.render("AI PILOT", True, (255, 255, 100))
        screen.blit(ai_label, (WIDTH // 2 - ai_label.get_width() // 2, 15))

        actions = []
        turn_v = a_turn.item()
        if turn_v == 0:
            actions.append("LEFT")
        elif turn_v == 2:
            actions.append("RIGHT")
        if a_thrust.item() == 1:
            actions.append("THRUST")
        act_str = " + ".join(actions) if actions else "COAST"
        act_txt = font.render(act_str, True, WHITE)
        screen.blit(act_txt, (WIDTH // 2 - act_txt.get_width() // 2, 40))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
