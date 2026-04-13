"""CNN+GRU policy for the drone intercept game.

Observation is a *single* 1D camera line (CAM_WIDTH) plus an IMU aux
vector (gyro_z, accel_fwd, accel_lat). A GRU core carries temporal
context across environment steps, so the policy can remember where the
target last was even when it leaves the camera FOV — fixing the
"overshoots and flails" failure mode a feed-forward policy suffers
when the target is not currently visible.

Outputs:
  - left_power  ~ Beta(α, β),  in [0, 1]
  - right_power ~ Beta(α, β),  in [0, 1]
  - value (1):  state value estimate
  - GRU hidden state (carried across steps)

The Beta heads emit (α, β) via softplus+1, guaranteeing α, β ≥ 1 so the
distribution is unimodal and log-densities at the endpoints stay bounded.

Two forward paths:
  * `step(...)`            — single-step, used for online rollout and the
                              in-game autopilot.
  * `evaluate_sequence(...)` — unrolls a whole (T, B, ...) rollout through
                              the GRU, zeroing the hidden state at episode
                              boundaries. Used by PPO updates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

CAM_WIDTH = 64
AUX_SIZE = 3
HIDDEN_SIZE = 256


class DronePolicy(nn.Module):
    def __init__(self):
        super().__init__()

        # ── 1D CNN encoder over a single camera line (B, 1, 64) ──────
        # Kernels chosen so we land on a (32, 8) feature map — same
        # 256-dim embedding the previous stacked-frame encoder produced.
        self.conv1 = nn.Conv1d(1, 16, kernel_size=8, stride=4, padding=2)   # → (B, 16, 16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=4, stride=2, padding=1)  # → (B, 32, 8)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1)  # → (B, 32, 8)

        self._conv_out_size = 32 * 8  # 256

        # ── Shared trunk → GRU ───────────────────────────────────────
        self.fc = nn.Linear(self._conv_out_size + AUX_SIZE, HIDDEN_SIZE)
        self.gru = nn.GRU(input_size=HIDDEN_SIZE, hidden_size=HIDDEN_SIZE,
                          num_layers=1)

        # ── Actor heads (Beta α, β per rotor) ────────────────────────
        self.left_head = nn.Linear(HIDDEN_SIZE, 2)
        self.right_head = nn.Linear(HIDDEN_SIZE, 2)

        # ── Critic head ──────────────────────────────────────────────
        self.value_head = nn.Linear(HIDDEN_SIZE, 1)

        # Orthogonal init for conv / fc / heads
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(m.bias)
        # Recurrent-specific init: orthogonal on weight_ih/hh, zero biases
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        # Smaller init for policy heads so the initial distribution is
        # close to Beta(softplus(0)+1, softplus(0)+1) ≈ Beta(1.69, 1.69).
        nn.init.orthogonal_(self.left_head.weight, gain=0.01)
        nn.init.orthogonal_(self.right_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    # ── Helpers ──────────────────────────────────────────────────────

    def initial_hidden(self, batch_size, device):
        """Return a zero GRU hidden state of shape (1, B, HIDDEN_SIZE)."""
        return torch.zeros(1, batch_size, HIDDEN_SIZE, device=device)

    def _encode(self, cam, aux):
        """Encode a flat batch of single-frame observations.

        cam: (N, CAM_WIDTH) float32
        aux: (N, AUX_SIZE) float32
        Returns features (N, HIDDEN_SIZE).
        """
        x = cam.unsqueeze(1)  # (N, 1, W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # (N, 256)
        x = torch.cat([x, aux], dim=1)
        x = F.relu(self.fc(x))
        return x

    def _beta_params(self, head_out):
        """Map raw head output (..., 2) to (α, β) with α, β ≥ 1."""
        ab = F.softplus(head_out) + 1.0
        return ab[..., 0], ab[..., 1]

    def _heads(self, features):
        """Apply the three heads to (N, HIDDEN_SIZE) features.

        Returns (alpha_l, beta_l), (alpha_r, beta_r), value.
        """
        alpha_l, beta_l = self._beta_params(self.left_head(features))
        alpha_r, beta_r = self._beta_params(self.right_head(features))
        value = self.value_head(features).squeeze(-1)
        return (alpha_l, beta_l), (alpha_r, beta_r), value

    # ── Single-step forward (online rollout / in-game autopilot) ─────

    def step(self, cam, aux, hidden,
             action_left=None, action_right=None):
        """One-step forward with recurrent state.

        cam:    (B, CAM_WIDTH) float32
        aux:    (B, AUX_SIZE) float32
        hidden: (1, B, HIDDEN_SIZE) GRU hidden state at this step
        action_left / action_right: optional float tensors in [0, 1]
            for off-policy evaluation; if None the policy samples.

        Returns action_left, action_right, log_prob, entropy, value,
                new_hidden (1, B, HIDDEN_SIZE).
        """
        feat = self._encode(cam, aux)  # (B, H)
        out, new_hidden = self.gru(feat.unsqueeze(0), hidden)  # (1, B, H)
        out = out.squeeze(0)  # (B, H)

        (alpha_l, beta_l), (alpha_r, beta_r), value = self._heads(out)

        left_dist = Beta(alpha_l, beta_l)
        right_dist = Beta(alpha_r, beta_r)

        if action_left is None:
            action_left = left_dist.sample()
            action_right = right_dist.sample()

        eps = 1e-6
        a_left = action_left.clamp(eps, 1.0 - eps)
        a_right = action_right.clamp(eps, 1.0 - eps)

        log_prob = left_dist.log_prob(a_left) + right_dist.log_prob(a_right)
        entropy = left_dist.entropy() + right_dist.entropy()

        return action_left, action_right, log_prob, entropy, value, new_hidden

    # ── Sequence forward (PPO update) ────────────────────────────────

    def evaluate_sequence(self, cam_seq, aux_seq, dones_seq, h_init,
                          action_left_seq, action_right_seq):
        """Unroll a (T, B, ...) rollout through the GRU, reset hidden at
        episode boundaries, and return per-timestep log-probs, entropy,
        and value estimates — all shape (T * B,).

        cam_seq:          (T, B, CAM_WIDTH)
        aux_seq:          (T, B, AUX_SIZE)
        dones_seq:        (T, B) float32 — 1.0 where the episode ended at
                          timestep t (so hidden state resets before t+1)
        h_init:           (1, B, HIDDEN_SIZE) hidden *before* timestep 0
        action_left_seq:  (T, B)
        action_right_seq: (T, B)
        """
        T, B = cam_seq.shape[:2]

        # Encode all timesteps in one fused batch — cheap and avoids a
        # Python-level loop over the (expensive) conv stack.
        feat = self._encode(
            cam_seq.reshape(T * B, -1),
            aux_seq.reshape(T * B, -1),
        ).view(T, B, HIDDEN_SIZE)

        # Unroll the GRU one step at a time so we can zero the hidden
        # state at episode boundaries. Horizons are small (~128) so the
        # Python loop overhead is negligible vs. the conv+linear cost.
        h = h_init
        outputs = []
        for t in range(T):
            step_out, h = self.gru(feat[t:t + 1], h)   # (1, B, H)
            outputs.append(step_out.squeeze(0))
            # If this step was terminal, reset hidden before step t+1.
            mask = (1.0 - dones_seq[t]).view(1, B, 1)
            h = h * mask

        features = torch.stack(outputs, dim=0)  # (T, B, H)
        flat = features.reshape(T * B, HIDDEN_SIZE)

        (alpha_l, beta_l), (alpha_r, beta_r), value = self._heads(flat)

        left_dist = Beta(alpha_l, beta_l)
        right_dist = Beta(alpha_r, beta_r)

        eps = 1e-6
        a_left = action_left_seq.reshape(-1).clamp(eps, 1.0 - eps)
        a_right = action_right_seq.reshape(-1).clamp(eps, 1.0 - eps)

        log_prob = left_dist.log_prob(a_left) + right_dist.log_prob(a_right)
        entropy = left_dist.entropy() + right_dist.entropy()

        return log_prob, entropy, value
