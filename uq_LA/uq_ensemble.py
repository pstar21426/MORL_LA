# Ensemble DQN as in Charpentier et al. (arXiv:2206.01558), sec. 4.
# K independent Gaussian Q-networks. Each outputs (mu, sigma) per action.
# Train each with Gaussian NLL on the DQN target
#   y = (r - rho * tau) + gamma**tau * max_a' mu_target(s', a').
# At decision time the policy maximises the ensemble mean mu.
# u_alea = mean_k sigma_k
# u_epist = mean_k (mu_k - mean_j mu_j)^2

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from uq_dqn import ReplayBuffer, Transition


class GaussianQNet(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, n_actions)
        self.log_std = nn.Linear(hidden, n_actions)
        nn.init.zeros_(self.log_std.weight)
        nn.init.zeros_(self.log_std.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(-4.0, 2.0)
        return mu, log_std


class EnsembleDQNAgent:
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        *,
        n_models: int = 5,
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
        seed: int = 0,
        device: Optional[str] = None,
    ):
        self.n_actions = int(n_actions)
        self.n_models = int(n_models)
        self.gamma = float(gamma)
        self.rho = rho
        self.batch_size = int(batch_size)
        self.target_sync = int(target_sync)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay_steps = int(epsilon_decay_steps)
        self.total_steps = 0
        self.last_nll = 0.0
        self.last_u_alea = 0.0
        self.last_u_epist = 0.0

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.nets: List[GaussianQNet] = []
        self.targets: List[GaussianQNet] = []
        self.opts: List[optim.Optimizer] = []
        for k in range(self.n_models):
            torch.manual_seed(int(seed) + 1000 * (k + 1))
            net = GaussianQNet(state_dim, n_actions, hidden=hidden).to(self.device)
            tgt = GaussianQNet(state_dim, n_actions, hidden=hidden).to(self.device)
            tgt.load_state_dict(net.state_dict())
            tgt.eval()
            self.nets.append(net)
            self.targets.append(tgt)
            self.opts.append(optim.Adam(net.parameters(), lr=lr))
        torch.manual_seed(int(seed))

        self.buffer = ReplayBuffer(buffer_size)
        self.cqi_visits = np.zeros(16, dtype=np.int64)

    @property
    def epsilon(self) -> float:
        t = min(self.total_steps, self.epsilon_decay_steps)
        frac = t / max(self.epsilon_decay_steps, 1)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def _mean_mu(self, state: np.ndarray) -> torch.Tensor:
        s = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(self.device)
        mus = []
        with torch.no_grad():
            for net in self.nets:
                mu, _ = net(s)
                mus.append(mu.squeeze(0))
        return torch.stack(mus, dim=0).mean(dim=0)

    def uncertainty_at(
        self, state: np.ndarray, action: Optional[int] = None
    ) -> Tuple[float, float, float]:
        s = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(self.device)
        mus, sigs = [], []
        with torch.no_grad():
            for net in self.nets:
                mu, log_std = net(s)
                mus.append(mu.squeeze(0))
                sigs.append(log_std.squeeze(0).exp())
        mu_k = torch.stack(mus, dim=0)
        sig_k = torch.stack(sigs, dim=0)
        mean_mu = mu_k.mean(dim=0)
        if action is None:
            action = int(mean_mu.argmax().item())
        u_alea = float(sig_k[:, action].mean().item())
        u_epist = float((mu_k[:, action] - mean_mu[action]).pow(2).mean().item())
        return float(mean_mu[action].item()), u_alea, u_epist

    def select_action(self, state: np.ndarray, *, greedy: bool = False) -> int:
        if not greedy and np.random.random() < self.epsilon:
            return int(np.random.randint(0, self.n_actions))
        q = self._mean_mu(state)
        return int(q.argmax().item())

    def push(self, tr: Transition) -> None:
        self.buffer.push(tr)
        cqi = int(np.clip(tr.cqi, 0, 15))
        self.cqi_visits[cqi] += 1

    def update(self) -> Optional[float]:
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
        shaped = rewards_t if self.rho is None else rewards_t - float(self.rho) * slots_t
        act = actions_t.unsqueeze(1)

        nlls = []
        mu_taken = []
        sig_taken = []
        sync = self.total_steps % self.target_sync == 0
        for net, tgt, opt in zip(self.nets, self.targets, self.opts):
            mu, log_std = net(states_t)
            mu_a = mu.gather(1, act).squeeze(1)
            log_std_a = log_std.gather(1, act).squeeze(1)
            with torch.no_grad():
                mu_next, _ = tgt(next_states_t)
                next_q = mu_next.max(dim=1).values
                target = shaped + discount_t * next_q * (1.0 - dones_t)
            var = torch.exp(2.0 * log_std_a).clamp_min(1e-8)
            nll = 0.5 * (2.0 * log_std_a + (target - mu_a).pow(2) / var)
            loss = nll.mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            if sync:
                tgt.load_state_dict(net.state_dict())
            nlls.append(float(loss.item()))
            mu_taken.append(mu_a.detach())
            sig_taken.append(log_std_a.detach().exp())

        mu_stack = torch.stack(mu_taken, dim=0)
        sig_stack = torch.stack(sig_taken, dim=0)
        mean_mu = mu_stack.mean(dim=0)
        self.last_nll = float(np.mean(nlls))
        self.last_u_alea = float(sig_stack.mean().item())
        self.last_u_epist = float((mu_stack - mean_mu).pow(2).mean().item())
        return self.last_nll

    def train_step(self) -> None:
        self.total_steps += 1
        self.update()

    def state_dict(self) -> dict:
        return {
            "nets": [net.state_dict() for net in self.nets],
            "cqi_visits": self.cqi_visits.copy(),
            "n_models": self.n_models,
        }
