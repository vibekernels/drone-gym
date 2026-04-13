#!/usr/bin/env python3
"""Recurrent PPO training for the drone intercept game.

PufferLib-inspired architecture:
  - Vectorized environments with shared numpy buffers
  - GRU-backed policy carrying temporal state across env steps
  - Segment-based experience storage
  - GAE advantage estimation
  - Clipped PPO objective with value clipping
  - Minibatches by env-sequence (not flat timesteps), so the GRU can
    unroll in order while computing PPO losses
"""

import os
# Beta.rsample/log_prob route through aten::_sample_dirichlet, which has no
# MPS kernel as of torch 2.x. Fall back to CPU for that one op so the rest
# of the policy/rollout still runs on the GPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
import time
import argparse

_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.log")
_log_fh = open(_LOG_PATH, "w")
_builtin_print = print
def print(*a, **kw):
    _builtin_print(*a, **kw)
    _log_fh.write(" ".join(str(x) for x in a) + kw.get("end", "\n"))
    _log_fh.flush()

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env import VecEnv, CAM_WIDTH
from policy import DronePolicy, AUX_SIZE, HIDDEN_SIZE


def parse_args():
    p = argparse.ArgumentParser(description="Train drone intercept agent with PPO")
    # Environment
    p.add_argument("--num-envs", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    # Rollout
    p.add_argument("--horizon", type=int, default=128,
                   help="Steps per env per rollout")
    # PPO
    p.add_argument("--epochs", type=int, default=4,
                   help="PPO epochs per rollout")
    p.add_argument("--num-minibatches", type=int, default=8,
                   help="Number of recurrent minibatches per epoch. "
                        "Each minibatch is a group of whole env-sequences "
                        "of length `horizon`, shuffled over env index.")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=2.5e-4)
    # Training
    p.add_argument("--total-timesteps", type=int, default=2_000_000)
    p.add_argument("--save-interval", type=int, default=50,
                   help="Save checkpoint every N updates")
    p.add_argument("--log-interval", type=int, default=5)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from")
    return p.parse_args()


@torch.no_grad()
def compute_gae(rewards, values, dones, next_value, gamma, gae_lambda):
    """Compute Generalized Advantage Estimation.

    All inputs: (horizon, num_envs)
    Returns advantages and returns, same shape.
    """
    horizon = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = 0

    for t in reversed(range(horizon)):
        if t == horizon - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]

        next_non_terminal = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

    returns = advantages + values
    return advantages, returns


def train():
    args = parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Environment ──────────────────────────────────────────────────
    vec_env = VecEnv(args.num_envs, seed=args.seed)
    obs_cam, obs_aux = vec_env.reset()

    batch_size = args.num_envs * args.horizon
    assert args.num_envs % args.num_minibatches == 0, (
        "num_envs must be divisible by num_minibatches for recurrent PPO")
    envs_per_mb = args.num_envs // args.num_minibatches
    num_updates = args.total_timesteps // batch_size

    print(f"Envs: {args.num_envs}  Horizon: {args.horizon}  "
          f"Batch: {batch_size}  Minibatches: {args.num_minibatches}  "
          f"Envs/mb: {envs_per_mb}")
    print(f"Total updates: {num_updates}  Total timesteps: {num_updates * batch_size}")

    # ── Policy & optimizer ───────────────────────────────────────────
    policy = DronePolicy().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr, eps=1e-5)

    global_step = 0
    start_update = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt["policy"])
        optimizer.load_state_dict(ckpt["optimizer"])
        global_step = ckpt.get("global_step", 0)
        start_update = ckpt.get("update", 0)
        print(f"Resumed from {args.resume} at step {global_step}")

    # ── Rollout buffers (horizon, num_envs, ...) ─────────────────────
    buf_cam = torch.zeros((args.horizon, args.num_envs, CAM_WIDTH),
                          dtype=torch.float32)
    buf_aux = torch.zeros((args.horizon, args.num_envs, AUX_SIZE),
                          dtype=torch.float32)
    buf_actions_left = torch.zeros((args.horizon, args.num_envs), dtype=torch.float32)
    buf_actions_right = torch.zeros((args.horizon, args.num_envs), dtype=torch.float32)
    buf_logprobs = torch.zeros((args.horizon, args.num_envs), dtype=torch.float32)
    buf_rewards = torch.zeros((args.horizon, args.num_envs), dtype=torch.float32)
    buf_dones = torch.zeros((args.horizon, args.num_envs), dtype=torch.float32)
    buf_values = torch.zeros((args.horizon, args.num_envs), dtype=torch.float32)

    # Persistent GRU hidden state carried across rollouts.
    next_hidden = policy.initial_hidden(args.num_envs, device)  # (1, N, H)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    start_time = time.time()

    for update in range(start_update, num_updates):
        # ── Learning rate annealing ──────────────────────────────────
        frac = 1.0 - update / num_updates
        lr_now = frac * args.lr
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        # Snapshot the hidden state at the *start* of this rollout — the
        # PPO update needs it to replay the GRU unroll from the same
        # initial condition the rollout used.
        initial_hidden_snapshot = next_hidden.detach().clone()  # (1, N, H)

        # ── Collect rollout ──────────────────────────────────────────
        policy.eval()
        for step in range(args.horizon):
            cam_t = torch.from_numpy(obs_cam).to(device)
            aux_t = torch.from_numpy(obs_aux).to(device)

            with torch.no_grad():
                a_left, a_right, logprob, _, value, next_hidden = policy.step(
                    cam_t, aux_t, next_hidden
                )

            buf_cam[step] = cam_t.cpu()
            buf_aux[step] = aux_t.cpu()
            buf_actions_left[step] = a_left.cpu()
            buf_actions_right[step] = a_right.cpu()
            buf_logprobs[step] = logprob.cpu()
            buf_values[step] = value.cpu()

            # Step environments
            obs_cam, obs_aux, rewards, dones, truncs = vec_env.step(
                a_left.cpu().numpy(), a_right.cpu().numpy()
            )

            buf_rewards[step] = torch.from_numpy(rewards)
            # For GAE, treat both terminal and truncated as episode boundaries
            done_mask_np = dones | truncs
            buf_dones[step] = torch.from_numpy(done_mask_np).float()

            # Reset hidden state for envs that just terminated so step
            # t+1 starts from a fresh zero state — matches the mask the
            # PPO update will apply inside evaluate_sequence.
            done_mask_t = torch.from_numpy(done_mask_np.astype(np.float32)).to(device)
            next_hidden = next_hidden * (1.0 - done_mask_t).view(1, -1, 1)

            global_step += args.num_envs

        # Bootstrap value for last step — uses next_hidden as it is,
        # post-done-masking from the final step.
        with torch.no_grad():
            next_cam = torch.from_numpy(obs_cam).to(device)
            next_aux = torch.from_numpy(obs_aux).to(device)
            # We don't want this call to advance next_hidden for the next
            # rollout, so we run it on a throwaway copy.
            _, _, _, _, next_value, _ = policy.step(
                next_cam, next_aux, next_hidden
            )
            next_value = next_value.cpu()

        # ── GAE ──────────────────────────────────────────────────────
        advantages, returns = compute_gae(
            buf_rewards, buf_values, buf_dones,
            next_value, args.gamma, args.gae_lambda
        )

        # ── PPO update: recurrent minibatches over envs ──────────────
        policy.train()
        clip_fracs = []
        total_pg_loss = 0.0
        total_v_loss = 0.0
        total_ent_loss = 0.0
        approx_kl = torch.tensor(0.0)

        for epoch in range(args.epochs):
            env_perm = torch.randperm(args.num_envs)

            for mb_start in range(0, args.num_envs, envs_per_mb):
                mb_env_ids = env_perm[mb_start:mb_start + envs_per_mb]

                mb_cam = buf_cam[:, mb_env_ids].to(device)              # (T, mb, W)
                mb_aux = buf_aux[:, mb_env_ids].to(device)              # (T, mb, A)
                mb_dones = buf_dones[:, mb_env_ids].to(device)          # (T, mb)
                mb_act_left = buf_actions_left[:, mb_env_ids].to(device)
                mb_act_right = buf_actions_right[:, mb_env_ids].to(device)
                mb_logprobs = buf_logprobs[:, mb_env_ids].reshape(-1).to(device)
                mb_values = buf_values[:, mb_env_ids].reshape(-1).to(device)
                mb_advantages = advantages[:, mb_env_ids].reshape(-1).to(device)
                mb_returns = returns[:, mb_env_ids].reshape(-1).to(device)
                mb_h_init = initial_hidden_snapshot[:, mb_env_ids]       # (1, mb, H)

                # Normalize advantages (over the whole minibatch: T*mb)
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Recurrent forward pass — unrolls the GRU through the
                # full horizon for this minibatch's envs.
                new_logprob, entropy, new_value = policy.evaluate_sequence(
                    mb_cam, mb_aux, mb_dones, mb_h_init,
                    mb_act_left, mb_act_right,
                )  # all (T*mb,)

                # Policy loss (clipped)
                log_ratio = new_logprob - mb_logprobs
                ratio = log_ratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clip_fracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (clipped)
                v_clipped = mb_values + torch.clamp(
                    new_value - mb_values, -args.clip_coef, args.clip_coef
                )
                v_loss1 = (new_value - mb_returns) ** 2
                v_loss2 = (v_clipped - mb_returns) ** 2
                v_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()

                ent_loss = entropy.mean()

                loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * ent_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()

                total_pg_loss += pg_loss.item()
                total_v_loss += v_loss.item()
                total_ent_loss += ent_loss.item()

        n_updates_done = args.epochs * args.num_minibatches
        avg_pg = total_pg_loss / n_updates_done
        avg_vl = total_v_loss / n_updates_done
        avg_ent = total_ent_loss / n_updates_done

        # ── Logging ──────────────────────────────────────────────────
        if (update + 1) % args.log_interval == 0:
            elapsed = time.time() - start_time
            sps = global_step / elapsed
            ep_rets, ep_lens = vec_env.pop_completed_episodes()

            log = (f"update {update + 1}/{num_updates}  "
                   f"step {global_step:,}  "
                   f"SPS {sps:.0f}  "
                   f"pg {avg_pg:.4f}  vl {avg_vl:.4f}  ent {avg_ent:.3f}  "
                   f"kl {approx_kl:.4f}  clip {np.mean(clip_fracs):.3f}  "
                   f"lr {lr_now:.2e}")

            if ep_rets:
                log += (f"  | ep_ret {np.mean(ep_rets):.2f} "
                        f"(max {np.max(ep_rets):.2f})  "
                        f"ep_len {np.mean(ep_lens):.0f}  "
                        f"n_eps {len(ep_rets)}")
            print(log)

        # ── Checkpoint ───────────────────────────────────────────────
        if (update + 1) % args.save_interval == 0 or update == num_updates - 1:
            path = os.path.join(args.checkpoint_dir, f"policy_{global_step}.pt")
            torch.save({
                "policy": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
                "update": update + 1,
                "args": vars(args),
            }, path)
            print(f"  → saved {path}")

    # Save final
    path = os.path.join(args.checkpoint_dir, "policy_final.pt")
    torch.save({
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
        "update": num_updates,
        "args": vars(args),
    }, path)
    print(f"Training complete. Final model: {path}")


if __name__ == "__main__":
    train()
