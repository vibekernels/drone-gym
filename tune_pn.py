#!/usr/bin/env python3
"""Parallel parameter tuning for the PN intercept controller.

All candidate configs run simultaneously inside a single VecEnv with
the vectorised BatchPNController.  Two-phase search:
  1. Coarse random sweep
  2. Refinement (perturbations around the best configs from phase 1)

Usage:
    python tune_pn.py                   # full search, saves pn_params.json
    python tune_pn.py --coarse 200      # more coarse samples
    python tune_pn.py --envs 64         # more envs per config
"""

import argparse
import time

import numpy as np
from dataclasses import asdict

from env import VecEnv
from pn_controller import PNParams, BatchPNController, save_params


# ── Defaults ────────────────────────────────────────────────────────────

ENVS_PER_CONFIG = 32          # parallel envs per parameter config
MAX_EPISODES    = 5           # episodes each env completes before we stop
MAX_STEPS_PER_EP = 600        # env horizon (10 s at 60 fps)

PARAM_RANGES = {
    "N":                    (2.0, 6.0),
    "K_bearing":            (0.5, 10.0),
    "K_rate":               (0.03, 0.5),
    "thrust_approach":      (0.4, 1.0),
    "thrust_min":           (0.15, 0.5),
    "K_accel":              (0.0, 2.0),
    "launch_min_brightness":(0.03, 0.25),
    "search_yaw":           (0.05, 0.4),
    "prescan_thrust":       (0.0, 0.25),
    "prescan_yaw":          (0.0, 0.15),
}


def random_params(rng):
    """Sample uniformly within each parameter range."""
    return PNParams(**{k: float(rng.uniform(*v)) for k, v in PARAM_RANGES.items()})


def perturb_params(base, rng, scale=0.12):
    """Gaussian perturbation around a base config, clamped to valid range."""
    d = asdict(base)
    for k, (lo, hi) in PARAM_RANGES.items():
        d[k] = float(np.clip(d[k] + rng.normal(0, scale * (hi - lo)), lo, hi))
    return PNParams(**d)


# ── Evaluation ──────────────────────────────────────────────────────────

def evaluate_batch(params_list, envs_per_config, max_episodes, seed=0):
    """Run all configs in one VecEnv; return per-config stats dicts."""
    n_cfg = len(params_list)
    total_envs = n_cfg * envs_per_config

    vec  = VecEnv(num_envs=total_envs, seed=seed)
    ctrl = BatchPNController(params_list, envs_per_config)

    obs_cam, obs_aux = vec.reset()

    hits       = np.zeros(total_envs, dtype=np.int32)
    episodes   = np.zeros(total_envs, dtype=np.int32)
    ep_reward  = np.zeros(total_envs)
    reward_sum = np.zeros(total_envs)

    budget = max_episodes * MAX_STEPS_PER_EP
    for _ in range(budget):
        left, right = ctrl.act(obs_cam, obs_aux)
        obs_cam, obs_aux, rewards, dones, truncs = vec.step(left, right)

        ep_reward += rewards
        resets = dones | truncs

        if resets.any():
            hits[dones & resets] += 1          # done = collision = hit
            episodes[resets] += 1
            reward_sum[resets] += ep_reward[resets]
            ep_reward[resets] = 0.0
            ctrl.reset_envs(resets)

        if episodes.min() >= max_episodes:
            break

    results = []
    for i in range(n_cfg):
        sl = slice(i * envs_per_config, (i + 1) * envs_per_config)
        h = int(hits[sl].sum())
        e = max(int(episodes[sl].sum()), 1)
        r = float(reward_sum[sl].sum()) / e
        results.append(dict(hit_rate=h / e, avg_reward=r,
                            total_hits=h, total_episodes=e))
    return results


# ── Main ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Tune PN controller params")
    ap.add_argument("--coarse",  type=int, default=120,
                    help="Number of random configs in phase 1")
    ap.add_argument("--refine",  type=int, default=80,
                    help="Number of perturbation configs in phase 2")
    ap.add_argument("--envs",    type=int, default=ENVS_PER_CONFIG,
                    help="Parallel envs per config")
    ap.add_argument("--episodes",type=int, default=MAX_EPISODES,
                    help="Episodes per env before scoring")
    ap.add_argument("--batch",   type=int, default=20,
                    help="Configs per VecEnv batch (memory vs. speed)")
    ap.add_argument("--seed",    type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    # ── Phase 1: coarse random sweep ────────────────────────────────
    print(f"Phase 1: {args.coarse} random configs  "
          f"({args.envs} envs x {args.episodes} eps each)")

    all_results = []
    for start in range(0, args.coarse, args.batch):
        end = min(start + args.batch, args.coarse)
        batch_params = [random_params(rng) for _ in range(end - start)]
        batch_stats  = evaluate_batch(batch_params, args.envs,
                                      args.episodes, seed=start * 1000)
        for p, s in zip(batch_params, batch_stats):
            all_results.append((p, s))

        best = max(all_results, key=lambda x: (x[1]["hit_rate"],
                                                x[1]["avg_reward"]))
        print(f"  [{end:3d}/{args.coarse}]  {time.time()-t0:5.1f}s  "
              f"best hit={best[1]['hit_rate']:.3f}  "
              f"rew={best[1]['avg_reward']:.1f}")

    all_results.sort(key=lambda x: (x[1]["hit_rate"],
                                     x[1]["avg_reward"]), reverse=True)
    print("\nTop 5 (coarse):")
    for i, (p, s) in enumerate(all_results[:5]):
        print(f"  #{i+1}  hit={s['hit_rate']:.3f}  rew={s['avg_reward']:+6.1f}  "
              f"N={p.N:.2f} Kb={p.K_bearing:.2f} Kr={p.K_rate:.3f} "
              f"thr={p.thrust_approach:.2f}/{p.thrust_min:.2f} "
              f"launch={p.launch_min_brightness:.3f} "
              f"search={p.search_yaw:.3f}")

    # ── Phase 2: refine around top configs ──────────────────────────
    top_k = min(3, len(all_results))
    bases = [r[0] for r in all_results[:top_k]]

    print(f"\nPhase 2: {args.refine} perturbations around top {top_k}")
    refine_results = list(all_results[:top_k])

    for start in range(0, args.refine, args.batch):
        end = min(start + args.batch, args.refine)
        batch_params = [perturb_params(bases[j % top_k], rng)
                        for j in range(end - start)]
        batch_stats = evaluate_batch(batch_params, args.envs,
                                     args.episodes,
                                     seed=(args.coarse + start) * 1000)
        for p, s in zip(batch_params, batch_stats):
            refine_results.append((p, s))

        best = max(refine_results, key=lambda x: (x[1]["hit_rate"],
                                                    x[1]["avg_reward"]))
        print(f"  [{end:3d}/{args.refine}]  {time.time()-t0:5.1f}s  "
              f"best hit={best[1]['hit_rate']:.3f}  "
              f"rew={best[1]['avg_reward']:.1f}")

    refine_results.sort(key=lambda x: (x[1]["hit_rate"],
                                        x[1]["avg_reward"]), reverse=True)

    best_params, best_stats = refine_results[0]

    print("\n" + "=" * 60)
    print("BEST CONFIG")
    print("=" * 60)
    print(f"  hit_rate : {best_stats['hit_rate']:.3f}  "
          f"({best_stats['total_hits']}/{best_stats['total_episodes']})")
    print(f"  reward   : {best_stats['avg_reward']:+.2f}")
    for k, v in asdict(best_params).items():
        print(f"  {k:25s}: {v:.4f}")

    save_params(best_params)
    print(f"\nSaved to pn_params.json  ({time.time()-t0:.1f}s total)")


if __name__ == "__main__":
    main()
