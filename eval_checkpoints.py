#!/usr/bin/env python3
"""Vectorized evaluation of one or more policy checkpoints.

Runs N envs in lockstep, records per-episode outcome, and reports
win rate (episodes that ended in a hit, not truncation).
"""

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import argparse
import numpy as np
import torch

from env import VecEnv, CAM_WIDTH, MAX_STEPS
from policy import DronePolicy, AUX_SIZE, HIDDEN_SIZE


@torch.no_grad()
def eval_ckpt(path, num_envs, num_episodes, device, deterministic=False):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    policy = DronePolicy().to(device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    vec = VecEnv(num_envs, seed=12345)
    obs_cam, obs_aux = vec.reset()
    hidden = policy.initial_hidden(num_envs, device)

    wins = 0
    completed = 0
    ep_returns_all = []
    ep_lengths_all = []

    while completed < num_episodes:
        cam_t = torch.from_numpy(obs_cam).to(device)
        aux_t = torch.from_numpy(obs_aux).to(device)
        feat = policy._encode(cam_t, aux_t)
        out, hidden = policy.gru(feat.unsqueeze(0), hidden)
        out = out.squeeze(0)
        (al, bl), (ar, br), _ = policy._heads(out)
        if deterministic:
            a_left = al / (al + bl)
            a_right = ar / (ar + br)
        else:
            from torch.distributions import Beta
            a_left = Beta(al, bl).sample()
            a_right = Beta(ar, br).sample()

        al_np = a_left.cpu().numpy()
        ar_np = a_right.cpu().numpy()
        obs_cam, obs_aux, rewards, dones, truncs = vec.step(al_np, ar_np)

        done_mask = dones | truncs
        for i in range(num_envs):
            if done_mask[i]:
                if dones[i]:
                    wins += 1
                completed += 1

        # Reset hidden state for completed envs (VecEnv auto-resets).
        if done_mask.any():
            mask_t = torch.from_numpy(done_mask.astype(np.float32)).to(device)
            hidden = hidden * (1.0 - mask_t).view(1, -1, 1)

    rets, lens = vec.pop_completed_episodes()
    # Trim to first `num_episodes` roughly — we might overshoot
    return {
        "path": path,
        "win_rate": wins / completed,
        "wins": wins,
        "completed": completed,
        "mean_ret": float(np.mean(rets)) if rets else 0.0,
        "mean_len": float(np.mean(lens)) if lens else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--num-episodes", type=int, default=500)
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    results = []
    for path in args.checkpoints:
        r = eval_ckpt(path, args.num_envs, args.num_episodes, device,
                      deterministic=args.deterministic)
        results.append(r)
        print(f"{os.path.basename(path):30s}  "
              f"wr {r['win_rate']*100:5.1f}%  "
              f"ret {r['mean_ret']:6.2f}  "
              f"len {r['mean_len']:6.0f}  "
              f"({r['wins']}/{r['completed']})")

    best = max(results, key=lambda x: x["win_rate"])
    print(f"\nBest: {os.path.basename(best['path'])}  "
          f"wr {best['win_rate']*100:.1f}%")


if __name__ == "__main__":
    main()
