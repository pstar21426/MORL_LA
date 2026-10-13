# DQN + disentangled uncertainty (Charpentier et al. style).
# - Policy: vanilla DQN or Double DQN on rate-shaped reward r - rho * tau.
# - Aleatoric / epistemic: evidential (NIG) regression on one-step target y = r - rho * tau
#   (no bootstrap on the uncertainty head — matches reward-uncertainty setup in the paper).

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SharedTrunk(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.out_dim = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QHead(nn.Module):
    def __init__(self, hidden: int, n_actions: int):
        super().__init__()
        self.fc = nn.Linear(hidden, n_actions)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc(h)


class EvidentialHead(nn.Module):
    """Per-action Normal-Inverse-Gamma parameters (gamma, nu, alpha, beta)."""

    def __init__(self, hidden: int, n_actions: int):
        super().__init__()
        self.fc = nn.Linear(hidden, n_actions * 4)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # (batch, n_actions, 4)
        b = h.shape[0]
        raw = self.fc(h).view(b, -1, 4)
        gamma = raw[..., 0]
        nu = F.softplus(raw[..., 1]) + 1.0
        alpha = F.softplus(raw[..., 2]) + 1.0
        beta = F.softplus(raw[..., 3]) + 1e-4
        return torch.stack([gamma, nu, alpha, beta], dim=-1)


def nig_gaussian_entropy(nu: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Entropy H(N(mu, sigma^2)) with sigma^2 = beta*(1+nu)/(nu*(alpha-1)) for NIG posterior mean."""
    var = beta * (1.0 + nu) / (nu * (alpha - 1.0) + 1e-6)
    return 0.5 * torch.log(2.0 * math.pi * math.e * var.clamp(min=1e-8))


def nig_posterior_entropy(nu: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """Approximate H(NIG) via differential entropy of scaled Inv-Gamma + Normal (Amini et al.)."""
    # Closed-form surrogate used in evidential RL repos: grows when nu, alpha small and beta large.
    return (
        0.5 * torch.log(math.pi / (nu + 1e-6))
        - alpha * torch.log(2.0 * beta + 1e-6)
        + (alpha + 0.5) * torch.log(1.0 + nu / (beta + 1e-6))
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )


def evidential_regression_loss(
    y: torch.Tensor,
    gamma: torch.Tensor,
    nu: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    *,
    reg_coeff: float = 0.1,
) -> torch.Tensor:
    two_bl = 2.0 * beta * (1.0 + nu)
    nll = (
        0.5 * torch.log(math.pi / (nu + 1e-6))
        - alpha * torch.log(two_bl + 1e-6)
        + (alpha + 0.5) * torch.log(nu * (y - gamma) ** 2 + two_bl + 1e-6)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )
    reg = torch.abs(y - gamma) * (2.0 * nu + alpha)
    return (nll + reg_coeff * reg).mean()


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminated: bool
    slots: int = 1
    cqi: int = 0


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buf)

    def push(self, tr: Transition) -> None:
        self.buf.append(tr)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        idx = np.random.choice(len(self.buf), batch_size, replace=False)
        batch = [self.buf[i] for i in idx]
        states = np.stack([np.asarray(t.state, dtype=np.float32) for t in batch])
        actions = np.asarray([t.action for t in batch], dtype=np.int64)
        rewards = np.asarray([t.reward for t in batch], dtype=np.float32)
        next_states = np.stack(
            [np.asarray(t.next_state, dtype=np.float32) for t in batch]
        )
        dones = np.asarray([t.terminated for t in batch], dtype=np.float32)
        slots = np.asarray([t.slots for t in batch], dtype=np.float32)
        cqis = np.asarray([t.cqi for t in batch], dtype=np.int64)
        return states, actions, rewards, next_states, dones, slots, cqis


class UncertaintyDQNAgent:
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        *,
        hidden: int = 256,
        gamma: float = 0.7,
        lr: float = 1e-3,
        buffer_size: int = 100_000,
        batch_size: int = 64,
        target_sync: int = 200,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 30_000,
        rho: Optional[float] = None,
        double_q: bool = True,
        evi_reg: float = 0.1,
        evi_loss_weight: float = 1.0,
        device: Optional[str] = None,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.rho = rho
        self.batch_size = batch_size
        self.target_sync = target_sync
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.double_q = double_q
        self.evi_reg = evi_reg
        self.evi_loss_weight = evi_loss_weight
        self.total_steps = 0
        self.last_q_loss = 0.0
        self.last_evi_loss = 0.0

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.trunk = SharedTrunk(state_dim, hidden=hidden).to(self.device)
        self.q_head = QHead(self.trunk.out_dim, n_actions).to(self.device)
        self.evi_head = EvidentialHead(self.trunk.out_dim, n_actions).to(self.device)

        self.trunk_t = SharedTrunk(state_dim, hidden=hidden).to(self.device)
        self.q_head_t = QHead(self.trunk_t.out_dim, n_actions).to(self.device)
        self._sync_target()

        params = (
            list(self.trunk.parameters())
            + list(self.q_head.parameters())
            + list(self.evi_head.parameters())
        )
        self.opt = optim.Adam(params, lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        # Epistemic reference: visit counts per CQI at decision time.
        self.cqi_visits = np.zeros(16, dtype=np.int64)

    def _sync_target(self) -> None:
        self.trunk_t.load_state_dict(self.trunk.state_dict())
        self.q_head_t.load_state_dict(self.q_head.state_dict())

    @property
    def epsilon(self) -> float:
        t = min(self.total_steps, self.epsilon_decay_steps)
        frac = t / max(self.epsilon_decay_steps, 1)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def _forward_q(self, states_t: torch.Tensor) -> torch.Tensor:
        h = self.trunk(states_t)
        return self.q_head(h)

    def _forward_evi(self, states_t: torch.Tensor) -> torch.Tensor:
        h = self.trunk(states_t)
        return self.evi_head(h)

    def uncertainty_at(
        self, state: np.ndarray, action: Optional[int] = None
    ) -> Tuple[float, float, float]:
        """Returns (mu_one_step, u_alea, u_epist) for action (default: greedy Q action)."""
        s = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self._forward_q(s).squeeze(0)
            if action is None:
                action = int(q.argmax().item())
            evi = self._forward_evi(s).squeeze(0)[action]
            gamma, nu, alpha, beta = evi[0], evi[1], evi[2], evi[3]
            u_alea = float(nig_gaussian_entropy(nu, alpha, beta).item())
            u_epist = float(nig_posterior_entropy(nu, alpha, beta).item())
            mu = float(gamma.item())
        return mu, u_alea, u_epist

    def select_action(self, state: np.ndarray, *, greedy: bool = False) -> int:
        if not greedy and np.random.random() < self.epsilon:
            return int(np.random.randint(0, self.n_actions))
        s = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self._forward_q(s)
        return int(q.argmax(dim=1).item())

    def push(self, tr: Transition) -> None:
        self.buffer.push(tr)
        cqi = int(np.clip(tr.cqi, 0, 15))
        self.cqi_visits[cqi] += 1

    def update(self) -> Optional[Tuple[float, float]]:
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones, slots, _cqis = self.buffer.sample(
            self.batch_size
        )
        states_t = torch.from_numpy(states).to(self.device)
        actions_t = torch.from_numpy(actions).to(self.device)
        rewards_t = torch.from_numpy(rewards).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        dones_t = torch.from_numpy(dones).to(self.device)
        slots_t = torch.from_numpy(slots).to(self.device)
        discount_t = self.gamma ** slots_t

        shaped = rewards_t
        if self.rho is not None:
            shaped = rewards_t - float(self.rho) * slots_t

        q_all = self._forward_q(states_t)
        q_sa = q_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.double_q:
                next_act = self._forward_q(next_states_t).argmax(dim=1)
                h_next = self.trunk_t(next_states_t)
                next_q = self.q_head_t(h_next).gather(1, next_act.unsqueeze(1)).squeeze(1)
            else:
                h_next = self.trunk_t(next_states_t)
                next_q = self.q_head_t(h_next).max(dim=1).values
            target_q = shaped + discount_t * next_q * (1.0 - dones_t)

        q_loss = F.mse_loss(q_sa, target_q)

        evi_all = self._forward_evi(states_t)
        evi_a = evi_all[torch.arange(actions_t.shape[0], device=self.device), actions_t]
        gamma, nu, alpha, beta = evi_a[:, 0], evi_a[:, 1], evi_a[:, 2], evi_a[:, 3]
        y_one = shaped.detach()
        evi_loss = evidential_regression_loss(
            y_one, gamma, nu, alpha, beta, reg_coeff=self.evi_reg
        )

        loss = q_loss + self.evi_loss_weight * evi_loss
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.trunk.parameters(), 5.0)
        self.opt.step()

        if self.total_steps % self.target_sync == 0:
            self._sync_target()

        self.last_q_loss = float(q_loss.item())
        self.last_evi_loss = float(evi_loss.item())
        return self.last_q_loss, self.last_evi_loss

    def train_step(self) -> None:
        self.total_steps += 1
        self.update()

    def state_dict(self) -> dict:
        return {
            "trunk": self.trunk.state_dict(),
            "q_head": self.q_head.state_dict(),
            "evi_head": self.evi_head.state_dict(),
            "cqi_visits": self.cqi_visits.copy(),
        }

    def load_state_dict(self, payload: dict) -> None:
        self.trunk.load_state_dict(payload["trunk"])
        self.q_head.load_state_dict(payload["q_head"])
        self.evi_head.load_state_dict(payload["evi_head"])
        if "cqi_visits" in payload:
            self.cqi_visits = np.asarray(payload["cqi_visits"], dtype=np.int64)
        self._sync_target()
