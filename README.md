# Drone Intercept 2D

A 2D arcade game where you pilot an interceptor drone to smash into a surveillance drone patrolling overhead. Includes a CNN-based reinforcement learning agent trained with PPO that achieves a 100% intercept rate.

## Gameplay

The target drone cruises back and forth at altitude, surveying the ground below. Your interceptor starts on the ground — use thrust and rotation to launch, find the target through your onboard camera, and ram it.

**Controls:**
- **Arrow keys / WASD** — turn left/right, thrust
- **C** — toggle AI autopilot
- **Space** — start / restart after a hit
- **Escape** — quit

## Architecture

### Game engine
- **Cython physics** (`physics.pyx`) — thrust, gravity, drag, collision detection, 1D camera projection
- **Pygame renderer** (`game.py`) — dynamic camera that zooms to keep both drones in frame, HUD with altimeter/speed/bearing, onboard camera view strip

### RL training
- **Environment** (`env.py`) — headless gym-style env with vectorized batch wrapper. Observation: 4 stacked 1D camera frames (4x64). Auxiliary input: IMU sensor (gyroscope + accelerometer in body frame)
- **Policy** (`policy.py`) — 2D CNN (Conv2d) encoder + shared MLP trunk, dual actor heads (turn 3-way + thrust 2-way), critic head. 44k parameters
- **Training** (`train.py`) — PufferLib-style PPO with GAE, clipped objectives, LR annealing, segment-based rollout buffers

### Key design choices
- **No privileged information** — the agent sees only what a real drone would: a forward-facing camera and IMU (gyro + accelerometer). No direct access to target position or distance
- **No world wrapping** — realistic physics where objects don't teleport across screen edges
- **Reward shaping** — large hit bonus (+50), small bearing/closing bonuses, time penalty. Tuned to avoid local optima from shaping rewards alone

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pygame-ce cython setuptools torch numpy gymnasium
python setup.py build_ext --inplace
```

## Usage

```bash
# Play the game
python game.py

# Train a new model (~22 min on CPU)
python train.py --num-envs 64 --horizon 128 --total-timesteps 2000000 --ent-coef 0.02

# Watch a trained model play
python enjoy.py checkpoints/policy_final.pt
```

## Evaluation

Trained for 2M steps (~22 min on Apple M-series CPU):

| Metric | Value |
|---|---|
| Hit rate | 100% (200/200) |
| Median intercept time | 4.3s |
| Fastest intercept | 2.8s |
| Mean return | 48.77 +/- 0.45 |
| Timeouts | 0% |
