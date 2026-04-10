"""CNN policy for the drone intercept game.

Observation is (NUM_FRAMES, CAM_WIDTH) = (4, 64) treated as a 2D image
with 1 channel for Conv2d.  IMU auxiliary vector (gyro_z, accel_fwd,
accel_lat) is concatenated after the conv encoder.

Outputs:
  - left_power  ~ Beta(α, β),  in [0, 1]   — left rotor throttle
  - right_power ~ Beta(α, β),  in [0, 1]   — right rotor throttle
  - value (1):  state value estimate

The Beta heads emit (α, β) via softplus+1, guaranteeing α, β ≥ 1 so the
distribution is unimodal and log-densities at the endpoints stay bounded.
At init the heads sit near Beta(~1.7, ~1.7), giving a gentle peak around
0.5 — a reasonable starting throttle for exploration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

NUM_FRAMES = 4
CAM_WIDTH = 64
AUX_SIZE = 3


class DronePolicy(nn.Module):
    def __init__(self):
        super().__init__()

        # ── 2D CNN encoder ───────────────────────────────────────────
        # Input: (B, 1, 4, 64)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(2, 8), stride=(1, 4), padding=(0, 2))
        # → (B, 16, 3, 16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(2, 4), stride=(1, 2), padding=(0, 1))
        # → (B, 32, 2, 8)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(2, 3), stride=(1, 1), padding=(0, 1))
        # → (B, 32, 1, 8)

        # Calculate flattened conv output size
        self._conv_out_size = 32 * 1 * 8  # 256

        # ── Shared trunk ─────────────────────────────────────────────
        self.fc = nn.Linear(self._conv_out_size + AUX_SIZE, 128)

        # ── Actor heads (Beta α, β per rotor) ────────────────────────
        self.left_head = nn.Linear(128, 2)   # → (alpha_raw, beta_raw)
        self.right_head = nn.Linear(128, 2)  # → (alpha_raw, beta_raw)

        # ── Critic head ──────────────────────────────────────────────
        self.value_head = nn.Linear(128, 1)

        # Orthogonal init (standard for PPO)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(m.bias)
        # Smaller init for policy heads so the initial distribution is
        # close to Beta(softplus(0)+1, softplus(0)+1) ≈ Beta(1.69, 1.69)
        nn.init.orthogonal_(self.left_head.weight, gain=0.01)
        nn.init.orthogonal_(self.right_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def encode(self, cam_obs, aux_obs):
        """Encode observations to hidden representation.

        cam_obs: (B, NUM_FRAMES, CAM_WIDTH) float32
        aux_obs: (B, AUX_SIZE) float32
        """
        x = cam_obs.unsqueeze(1)  # (B, 1, 4, 64)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # (B, 256)
        x = torch.cat([x, aux_obs], dim=1)  # (B, 259)
        x = F.relu(self.fc(x))  # (B, 128)
        return x

    def _beta_params(self, head_out):
        """Map raw head output (B, 2) to (α, β) with α, β ≥ 1."""
        ab = F.softplus(head_out) + 1.0
        return ab[..., 0], ab[..., 1]

    def forward(self, cam_obs, aux_obs):
        """Full forward pass. Returns (alpha_l, beta_l), (alpha_r, beta_r), value."""
        h = self.encode(cam_obs, aux_obs)
        alpha_l, beta_l = self._beta_params(self.left_head(h))
        alpha_r, beta_r = self._beta_params(self.right_head(h))
        value = self.value_head(h).squeeze(-1)
        return (alpha_l, beta_l), (alpha_r, beta_r), value

    def get_action_and_value(self, cam_obs, aux_obs,
                             action_left=None, action_right=None):
        """Sample or evaluate continuous actions. Used by PPO.

        action_left, action_right: float tensors in [0, 1] (or None to sample).

        Returns: action_left, action_right, log_prob, entropy, value
        """
        (alpha_l, beta_l), (alpha_r, beta_r), value = self.forward(cam_obs, aux_obs)

        left_dist = Beta(alpha_l, beta_l)
        right_dist = Beta(alpha_r, beta_r)

        if action_left is None:
            action_left = left_dist.sample()
            action_right = right_dist.sample()

        # Numerical safety: Beta log_prob is undefined exactly at 0 or 1.
        eps = 1e-6
        a_left = action_left.clamp(eps, 1.0 - eps)
        a_right = action_right.clamp(eps, 1.0 - eps)

        log_prob = left_dist.log_prob(a_left) + right_dist.log_prob(a_right)
        entropy = left_dist.entropy() + right_dist.entropy()

        return action_left, action_right, log_prob, entropy, value
