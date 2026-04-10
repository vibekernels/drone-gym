"""Benchmark CPU vs MPS for training throughput.

Runs a small number of PPO updates on each device and reports SPS.
"""

import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env import VecEnv, NUM_FRAMES, CAM_WIDTH
from policy import DronePolicy, AUX_SIZE


def run(device_str, num_updates=10, num_envs=64, horizon=128,
        minibatch_size=512, epochs=4):
    device = torch.device(device_str)
    torch.manual_seed(0)
    np.random.seed(0)

    vec_env = VecEnv(num_envs, seed=0)
    obs_cam, obs_aux = vec_env.reset()

    batch_size = num_envs * horizon
    num_minibatches = batch_size // minibatch_size

    policy = DronePolicy().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=2.5e-4, eps=1e-5)

    buf_cam = torch.zeros((horizon, num_envs, NUM_FRAMES, CAM_WIDTH), dtype=torch.float32)
    buf_aux = torch.zeros((horizon, num_envs, AUX_SIZE), dtype=torch.float32)
    buf_actions_turn = torch.zeros((horizon, num_envs), dtype=torch.long)
    buf_actions_thrust = torch.zeros((horizon, num_envs), dtype=torch.long)
    buf_logprobs = torch.zeros((horizon, num_envs), dtype=torch.float32)
    buf_values = torch.zeros((horizon, num_envs), dtype=torch.float32)
    buf_rewards = torch.zeros((horizon, num_envs), dtype=torch.float32)
    buf_dones = torch.zeros((horizon, num_envs), dtype=torch.float32)

    # Warmup (skip from timing)
    for _ in range(2):
        for step in range(horizon):
            cam_t = torch.from_numpy(obs_cam).to(device)
            aux_t = torch.from_numpy(obs_aux).to(device)
            with torch.no_grad():
                a_turn, a_thrust, logprob, _, value = policy.get_action_and_value(cam_t, aux_t)
            obs_cam, obs_aux, _, _, _ = vec_env.step(a_turn.cpu().numpy(), a_thrust.cpu().numpy())
        # One fake PPO epoch
        b_cam = torch.randn(batch_size, NUM_FRAMES, CAM_WIDTH, device=device)
        b_aux = torch.randn(batch_size, AUX_SIZE, device=device)
        b_at = torch.zeros(batch_size, dtype=torch.long, device=device)
        b_ath = torch.zeros(batch_size, dtype=torch.long, device=device)
        for start in range(0, batch_size, minibatch_size):
            mb = slice(start, start + minibatch_size)
            _, _, lp, ent, val = policy.get_action_and_value(b_cam[mb], b_aux[mb], b_at[mb], b_ath[mb])
            loss = (val.pow(2).mean() - lp.mean() - ent.mean())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if device_str == "mps":
        torch.mps.synchronize()

    # Timed loop
    t_start = time.time()
    total_steps = 0

    for update in range(num_updates):
        # Rollout
        for step in range(horizon):
            cam_t = torch.from_numpy(obs_cam).to(device)
            aux_t = torch.from_numpy(obs_aux).to(device)
            with torch.no_grad():
                a_turn, a_thrust, logprob, _, value = policy.get_action_and_value(cam_t, aux_t)
            buf_cam[step] = cam_t.cpu()
            buf_aux[step] = aux_t.cpu()
            buf_actions_turn[step] = a_turn.cpu()
            buf_actions_thrust[step] = a_thrust.cpu()
            buf_logprobs[step] = logprob.cpu()
            buf_values[step] = value.cpu()
            obs_cam, obs_aux, rewards, dones, truncs = vec_env.step(
                a_turn.cpu().numpy(), a_thrust.cpu().numpy()
            )
            buf_rewards[step] = torch.from_numpy(rewards)
            buf_dones[step] = torch.from_numpy(dones | truncs).float()
            total_steps += num_envs

        # PPO update (using random advantages for timing — we don't care about learning here)
        b_cam = buf_cam.reshape(-1, NUM_FRAMES, CAM_WIDTH)
        b_aux = buf_aux.reshape(-1, AUX_SIZE)
        b_at = buf_actions_turn.reshape(-1)
        b_ath = buf_actions_thrust.reshape(-1)
        b_lp = buf_logprobs.reshape(-1)
        b_adv = torch.randn(batch_size)
        b_ret = torch.randn(batch_size)
        b_val = buf_values.reshape(-1)

        for epoch in range(epochs):
            indices = torch.randperm(batch_size)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_idx = indices[start:end]
                mb_cam = b_cam[mb_idx].to(device)
                mb_aux = b_aux[mb_idx].to(device)
                mb_at = b_at[mb_idx].to(device)
                mb_ath = b_ath[mb_idx].to(device)
                mb_lp = b_lp[mb_idx].to(device)
                mb_adv = b_adv[mb_idx].to(device)
                mb_ret = b_ret[mb_idx].to(device)
                mb_val = b_val[mb_idx].to(device)

                _, _, new_lp, ent, new_v = policy.get_action_and_value(mb_cam, mb_aux, mb_at, mb_ath)
                ratio = (new_lp - mb_lp).exp()
                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 0.8, 1.2)
                pg = torch.max(pg1, pg2).mean()
                vl = 0.5 * ((new_v - mb_ret) ** 2).mean()
                loss = pg + 0.5 * vl - 0.01 * ent.mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()

    if device_str == "mps":
        torch.mps.synchronize()

    elapsed = time.time() - t_start
    sps = total_steps / elapsed
    return elapsed, sps, total_steps


def main():
    print("Warming up CPU...")
    t_cpu, sps_cpu, steps = run("cpu", num_updates=10)
    print(f"  CPU: {t_cpu:.2f}s for {steps:,} steps → {sps_cpu:.0f} SPS")

    if torch.backends.mps.is_available():
        print("Warming up MPS...")
        t_mps, sps_mps, steps = run("mps", num_updates=10)
        print(f"  MPS: {t_mps:.2f}s for {steps:,} steps → {sps_mps:.0f} SPS")
        print(f"\nSpeedup: {sps_mps / sps_cpu:.2f}x {'(MPS wins)' if sps_mps > sps_cpu else '(CPU wins)'}")
    else:
        print("MPS not available")


if __name__ == "__main__":
    main()
