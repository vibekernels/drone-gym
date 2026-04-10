"""CNN policy for the drone intercept game.

Observation is (NUM_FRAMES, CAM_WIDTH) = (4, 64) treated as a 2D image
with 1 channel for Conv2d.  Auxiliary vector is concatenated after the
conv encoder.

Outputs:
  - turn_logits (3): left / none / right
  - thrust_logits (2): off / on
  - value (1): state value estimate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

NUM_FRAMES = 4
CAM_WIDTH = 64
AUX_SIZE = 4


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

        # ── Actor heads ──────────────────────────────────────────────
        self.turn_head = nn.Linear(128, 3)    # left, none, right
        self.thrust_head = nn.Linear(128, 2)  # off, on

        # ── Critic head ──────────────────────────────────────────────
        self.value_head = nn.Linear(128, 1)

        # Orthogonal init (standard for PPO)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(m.bias)
        # Smaller init for policy heads
        nn.init.orthogonal_(self.turn_head.weight, gain=0.01)
        nn.init.orthogonal_(self.thrust_head.weight, gain=0.01)
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
        x = torch.cat([x, aux_obs], dim=1)  # (B, 260)
        x = F.relu(self.fc(x))  # (B, 128)
        return x

    def forward(self, cam_obs, aux_obs):
        """Full forward pass. Returns turn_logits, thrust_logits, value."""
        h = self.encode(cam_obs, aux_obs)
        return self.turn_head(h), self.thrust_head(h), self.value_head(h).squeeze(-1)

    def get_action_and_value(self, cam_obs, aux_obs, action_turn=None, action_thrust=None):
        """Sample or evaluate actions. Used by PPO.

        If actions are None, samples new ones.
        If actions are provided, evaluates log-probs and entropy for those actions.

        Returns: action_turn, action_thrust, log_prob, entropy, value
        """
        turn_logits, thrust_logits, value = self.forward(cam_obs, aux_obs)

        turn_dist = Categorical(logits=turn_logits)
        thrust_dist = Categorical(logits=thrust_logits)

        if action_turn is None:
            action_turn = turn_dist.sample()
            action_thrust = thrust_dist.sample()

        log_prob = turn_dist.log_prob(action_turn) + thrust_dist.log_prob(action_thrust)
        entropy = turn_dist.entropy() + thrust_dist.entropy()

        return action_turn, action_thrust, log_prob, entropy, value
