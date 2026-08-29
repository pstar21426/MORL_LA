# Minimal DQN for DownlinkLAEnv (decision-step, state-only policy)

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def preprocess_state(state: np.ndarray) -> np.ndarray:
    """Map unseen (-1) history slots to 0 for the network."""
    s = np.asarray(state, dtype=np.float32).copy()
    s[s < 0.0] = 0.0
    return s


class QNet(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


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
        states = np.stack([preprocess_state(t.state) for t in batch])
        actions = np.asarray([t.action for t in batch], dtype=np.int64)
        rewards = np.asarray([t.reward for t in batch], dtype=np.float32)
        next_states = np.stack([preprocess_state(t.next_state) for t in batch])
        dones = np.asarray([t.done for t in batch], dtype=np.float32)
        return states, actions, rewards, next_states, dones


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        *,
        hidden: int = 256,
        gamma: float = 0.99,
        lr: float = 1e-3,
        buffer_size: int = 20_000,
        batch_size: int = 64,
        target_sync: int = 200,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 5_000,
        device: Optional[str] = None,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_sync = target_sync
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.total_steps = 0

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.q = QNet(state_dim, n_actions, hidden=hidden).to(self.device)
        self.q_target = QNet(state_dim, n_actions, hidden=hidden).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

    @property
    def epsilon(self) -> float:
        t = min(self.total_steps, self.epsilon_decay_steps)
        frac = t / max(self.epsilon_decay_steps, 1)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def select_action(self, state: np.ndarray, *, greedy: bool = False) -> int:
        if not greedy and np.random.random() < self.epsilon:
            return int(np.random.randint(0, self.n_actions))
        s = torch.from_numpy(preprocess_state(state)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.q(s)
        return int(q.argmax(dim=1).item())

    def push(self, tr: Transition) -> None:
        self.buffer.push(tr)

    def update(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )
        states_t = torch.from_numpy(states).to(self.device)
        actions_t = torch.from_numpy(actions).to(self.device)
        rewards_t = torch.from_numpy(rewards).to(self.device)
        next_states_t = torch.from_numpy(next_states).to(self.device)
        dones_t = torch.from_numpy(dones).to(self.device)

        q_sa = self.q(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.q_target(next_states_t).max(dim=1).values
            target = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = nn.functional.mse_loss(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        if self.total_steps % self.target_sync == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        return float(loss.item())

    def train_step(self) -> None:
        self.total_steps += 1
        self.update()
