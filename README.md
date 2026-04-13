# Drone Intercept 2D

A 2D arcade game where you pilot an interceptor drone to smash into a surveillance drone patrolling overhead. Includes a Proportional Navigation (PN) guidance controller that achieves a 99.8% intercept rate using only onboard camera and IMU sensors.

## Gameplay

The target drone cruises back and forth at altitude across a wide patrol zone. Your interceptor starts on the ground — use independent rotor throttles to launch, find the target through your onboard camera, and ram it.

**Controls:**
- **Q/A/Z** — left rotor at 100/50/25%
- **E/D/C** — right rotor at 100/50/25%
- **P** — toggle AI autopilot (PN guidance)
- **Space** — start / restart after a hit
- **Escape** — quit

## Architecture

### Game engine
- **Cython physics** (`physics.pyx`) — rigid-body angular dynamics, independent rotor thrust, gravity, drag, collision detection, 1D camera projection
- **Pygame renderer** (`game.py`) — dynamic camera that zooms to keep both drones in frame, HUD with altimeter/speed/bearing, onboard camera view strip, PN guidance state panel

### PN guidance controller
- **Controller** (`pn_controller.py`) — camera+IMU proportional navigation with three phases:
  1. **Pre-launch scan** — hovers and rotates to find the target before committing
  2. **Track** — hybrid pursuit + PN guidance steers toward intercept, with IMU-based gravity compensation
  3. **Search** — rotates toward last-known bearing when target exits FOV
- **Tuning** (`tune_pn.py`) — parallel parameter search using headless vectorized simulations. Runs hundreds of parameter configs simultaneously via `BatchPNController`, with a two-phase coarse random sweep + refinement strategy

### Sensor model
- **Camera** — 64-pixel 1D forward-facing line sensor (90 deg FOV). Target appears as a bright bar whose position gives bearing and whose width gives range
- **IMU** — gyroscope (angular rate) + accelerometer (body-frame proper acceleration, excluding gravity)
- **No privileged information** — the controller sees only what a real drone would. Target bearing, LOS rate, and range are all derived from camera pixels and IMU readings

### Key design choices
- **No world wrapping** — realistic physics where objects don't teleport across screen edges
- **Independent rotor control** — differential thrust creates yaw torque with real angular inertia, combined thrust provides body-up force
- **Classical guidance** — PN is a proven missile guidance law that steers to null the line-of-sight rate, ensuring a collision course. The tuned parameters were found via automated search, not manual tuning

## Environment (`env.py`)
Headless gym-style environment with vectorized batch wrapper (`VecEnv`). Observation: single 1D camera line (64 px) + IMU aux vector (3). Action: continuous left/right rotor throttles in [0, 1].

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pygame cython setuptools numpy
python setup.py build_ext --inplace
```

## Usage

```bash
# Play the game (P to toggle AI autopilot)
python game.py

# Tune PN parameters (~6 min, saves pn_params.json)
python tune_pn.py

# Watch the PN controller play
python enjoy.py
```

## Evaluation

Tuned over 200 random + 120 refinement configs (48 envs x 8 episodes each):

| Metric | Value |
|---|---|
| Hit rate | 99.8% (499/500) |
| Median intercept time | 5.2s |
| Mean intercept time | 4.9s |
| P90 intercept time | 7.3s |
| Timeouts | 0.2% (1/500) |
