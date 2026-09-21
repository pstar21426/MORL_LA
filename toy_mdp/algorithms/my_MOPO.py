"""
MOPO for Appendix B toy LA — aleatoric overpessimism study.

Claim:
  vanilla (predicted reward variance) mixes ACK aleatoric into the penalty → collapses
  oracle  (= debias): u = relu(u_vanilla - Var_oracle[R|p])  → strips known aleatoric → survives

Modes:
  vanilla   — max predicted reward variance (bad arm)
  oracle    — alias of debias: remove oracle ACK variance from u (good arm / cheat)
  debias    — same as oracle
  oracle_p  — OPTIONAL ablation: penalty = Var_oracle only (should hurt; not the good arm)
  epistemic — ensemble mean disagreement only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.la_env import SCALE_ACTION, make_la_env

# --- hyperparameters (MOPO-style; scaled for LA toy runs) ---
DEFAULT_DATASET = ROOT / "datasets" / "la_dp_eps0.25_steps10000_ep1_seed0.npz"

LR_ACTOR = 3e-4
LR_CRITIC = 3e-4
LR_ALPHA = 3e-4
LR_DYNAMICS = 1e-3
GAMMA = 0.99
BATCH_SIZE = 256
TAU = 0.005

NUM_ENSEMBLE = 7
DYNAMICS_TRAIN_STEPS = 1000
DYNAMICS_HIDDEN = 200

ROLLOUT_INTERVAL = 1000
ROLLOUT_HORIZON = 1
ROLLOUT_BATCH_SIZE = 1000
PENALTY_COEF = 5.0
REAL_RATIO = 0.05

POLICY_TRAIN_STEPS = 3000
EVAL_FREQ = 3000
EVAL_EPISODES = 1
EVAL_MAX_STEPS = 1000

UNCERTAINTY_MODES = ("vanilla", "epistemic", "oracle", "oracle_p", "debias")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, max_size: int):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.not_done = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = np.asarray(action, dtype=np.float32).reshape(self.action_dim)
        self.reward[self.ptr] = np.asarray(reward, dtype=np.float32).reshape(1)
        self.next_state[self.ptr] = next_state
        done_f = float(np.asarray(done).reshape(-1)[0])
        self.not_done[self.ptr] = 1.0 - done_f

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_batch(self, states, actions, rewards, next_states, dones):
        for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
            self.add(s, a, r, ns, d)

    def sample(self, batch_size: int):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(device),
            torch.FloatTensor(self.action[ind]).to(device),
            torch.FloatTensor(self.reward[ind]).to(device),
            torch.FloatTensor(self.next_state[ind]).to(device),
            torch.FloatTensor(self.not_done[ind]).to(device),
        )

    def sample_states(self, batch_size: int):
        ind = np.random.randint(0, self.size, size=batch_size)
        return torch.FloatTensor(self.state[ind]).to(device)

    def normalize_states(self, eps: float = 1e-3):
        mean = self.state[: self.size].mean(0, keepdims=True)
        std = self.state[: self.size].std(0, keepdims=True) + eps
        self.state[: self.size] = (self.state[: self.size] - mean) / std
        self.next_state[: self.size] = (self.next_state[: self.size] - mean) / std
        return mean.astype(np.float32), std.astype(np.float32)

    def copy_from(self, other: "ReplayBuffer"):
        n = other.size
        self.state[:n] = other.state[:n]
        self.action[:n] = other.action[:n]
        self.reward[:n] = other.reward[:n]
        self.next_state[:n] = other.next_state[:n]
        self.not_done[:n] = other.not_done[:n]
        self.size = n
        self.ptr = n % self.max_size


def load_npz_buffer(path: Path, state_dim: int, action_dim: int) -> Tuple[ReplayBuffer, dict]:
    """Load Appendix B collect.py .npz into a ReplayBuffer."""
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    meta = json.loads(str(data["meta_json"]))

    obs = np.asarray(data["observations"], dtype=np.float32)
    next_obs = np.asarray(data["next_observations"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32).reshape(-1, action_dim)
    rewards = np.asarray(data["rewards"], dtype=np.float32).reshape(-1, 1)
    dones = np.asarray(data["terminations"], dtype=np.float32).reshape(-1, 1)

    if obs.shape[1] != state_dim:
        raise ValueError(f"obs dim {obs.shape[1]} != env state_dim {state_dim}")
    if actions.shape[1] != action_dim:
        raise ValueError(f"action dim {actions.shape[1]} != {action_dim}")

    buffer = ReplayBuffer(state_dim, action_dim, max_size=len(rewards))
    for i in range(len(rewards)):
        buffer.add(obs[i], actions[i], rewards[i], next_obs[i], float(dones[i, 0]))
    return buffer, meta


def continuous_to_discrete(action_cont, n_actions: int) -> np.ndarray:
    """Round continuous MCS index to Discrete gym action."""
    a = np.asarray(action_cont, dtype=np.float32).reshape(-1)
    return np.clip(np.rint(a), 0, n_actions - 1).astype(np.int64)


def continuous_to_discrete_torch(action_cont: torch.Tensor, max_action: float) -> torch.Tensor:
    """Round continuous actions for dynamics / buffer storage (still float)."""
    return torch.clamp(torch.round(action_cont), 0.0, max_action)


def denormalize_states(
    states_norm: torch.Tensor, state_mean: np.ndarray, state_std: np.ndarray
) -> torch.Tensor:
    mean = torch.as_tensor(state_mean, dtype=states_norm.dtype, device=states_norm.device)
    std = torch.as_tensor(state_std, dtype=states_norm.dtype, device=states_norm.device)
    return states_norm * std + mean


def oracle_aleatoric_from_norm_batch(
    states_norm: torch.Tensor,
    actions_gym: torch.Tensor,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    *,
    beta: float = 0.5,
    reward_scale: float = 1.0,
    n_actions: int = 28,
) -> torch.Tensor:
    """
    Oracle aleatoric reward variance from ACK/NACK Bernoulli:
      Var[R | k,x,a] = p(1-p) (r_ok - r_fail)^2
      p = tanh(x / (18 a)),  a = paper action
      r_ok = scale * tanh(a / n_actions)
      r_fail = scale * (-beta (k+1))
    """
    raw = denormalize_states(states_norm, state_mean, state_std)
    k = torch.clamp(raw[:, 0:1], min=0.0)
    context = torch.clamp(raw[:, 1:2], min=0.0)
    paper_a = torch.clamp(actions_gym, min=0.0) + 1.0
    p = torch.tanh(context / (SCALE_ACTION * paper_a))
    r_ok = float(reward_scale) * torch.tanh(paper_a / float(n_actions))
    r_fail = float(reward_scale) * (-float(beta) * (k + 1.0))
    return p * (1.0 - p) * (r_ok - r_fail).pow(2)


class EnsembleDynamics(nn.Module):
    """Probabilistic ensemble: each member predicts (delta_s, reward) with Gaussian NLL."""

    def __init__(self, state_dim, action_dim, num_models=NUM_ENSEMBLE, hidden=DYNAMICS_HIDDEN):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_models = num_models
        self.output_dim = state_dim + 1

        self.models = nn.ModuleList(
            [self._build_model(state_dim + action_dim, hidden, self.output_dim) for _ in range(num_models)]
        )

    @staticmethod
    def _build_model(in_dim, hidden, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim * 2),
        )

    def forward(self, state, action, model_idx=None):
        x = torch.cat([state, action], dim=-1)
        if model_idx is not None:
            out = self.models[model_idx](x)
            mean, logvar = torch.chunk(out, 2, dim=-1)
            return mean, logvar.clamp(-10.0, 0.5)

        means, logvars = [], []
        for model in self.models:
            out = model(x)
            mean, logvar = torch.chunk(out, 2, dim=-1)
            means.append(mean)
            logvars.append(logvar.clamp(-10.0, 0.5))
        return torch.stack(means), torch.stack(logvars)

    def predict_one(self, state, action, model_idx=None, sample=True):
        if model_idx is None:
            model_idx = np.random.randint(0, self.num_models)
        mean, logvar = self(state, action, model_idx=model_idx)
        if sample:
            pred = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        else:
            pred = mean
        delta_s = pred[:, : self.state_dim]
        reward = pred[:, self.state_dim : self.state_dim + 1]
        return delta_s, reward, model_idx

    def uncertainty_vanilla(self, state, action):
        """Max ensemble predicted variance on the reward head (ACK/NACK noise)."""
        means, logvars = self.forward(state, action)
        var = torch.exp(logvars)
        # last dim = reward; ignore context-x variance which is pure aleatoric i.i.d.
        rew_var = var[..., -1:]  # (E, B, 1)
        return rew_var.max(dim=0).values

    def uncertainty_epistemic(self, state, action):
        """Disagreement of ensemble mean reward predictions."""
        means, _ = self.forward(state, action)
        rew_means = means[..., -1:]  # (E, B, 1)
        return rew_means.std(dim=0)

    def uncertainty(
        self,
        state,
        action,
        mode: str = "vanilla",
        state_mean: Optional[np.ndarray] = None,
        state_std: Optional[np.ndarray] = None,
        reward_scale: float = 1.0,
        beta: float = 0.5,
        n_actions: int = 28,
        u_scale_van: float = 1.0,
        u_scale_orc: float = 1.0,
        u_scale_resid: float = 1.0,
        alpha_orc: float = 1.0,
        use_ep_fallback: bool = False,
    ):
        """
        Penalties are mean-normalized on real data so λ is comparable across modes.

        oracle/debias: residual after LS projection of pred-var onto oracle aleatoric
          u = relu(u_van - alpha * u_orc) / scale_resid
        If that residual is negligible vs ensemble disagreement, fall back to epistemic.
        """
        mode = mode.lower()
        if mode not in UNCERTAINTY_MODES:
            raise ValueError(f"unknown uncertainty mode {mode}; choose from {UNCERTAINTY_MODES}")

        eps = 1e-6
        scale_van = float(max(u_scale_van, eps))
        scale_orc = float(max(u_scale_orc, eps))
        scale_resid = float(max(u_scale_resid, eps))

        u_van_raw = self.uncertainty_vanilla(state, action)

        if mode == "vanilla":
            return u_van_raw / scale_van

        if mode == "epistemic":
            # scale_resid holds mean epistemic when calibrating for this mode
            return self.uncertainty_epistemic(state, action) / scale_resid

        if state_mean is None or state_std is None:
            raise ValueError(f"mode={mode} requires state_mean/state_std for oracle p")
        u_oracle_raw = oracle_aleatoric_from_norm_batch(
            state,
            action,
            state_mean,
            state_std,
            beta=beta,
            reward_scale=reward_scale,
            n_actions=n_actions,
        )

        if mode == "oracle_p":
            return u_oracle_raw / scale_orc

        if mode in ("oracle", "debias"):
            if use_ep_fallback:
                return self.uncertainty_epistemic(state, action) / scale_resid
            resid = torch.relu(u_van_raw - float(alpha_orc) * u_oracle_raw)
            return resid / scale_resid
        raise ValueError(f"unknown uncertainty mode {mode}")


@torch.no_grad()
def calibrate_penalty_scales(
    ensemble: EnsembleDynamics,
    real_buffer: ReplayBuffer,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    *,
    beta: float,
    reward_scale: float,
    n_actions: int,
    num_samples: int = 4096,
) -> dict:
    """
    Calibrate penalty scales on real (s,a):

    - scale_van / scale_orc: means of raw vanilla / oracle (unit-mean penalties)
    - alpha_orc: LS coeff minimizing ||u_van - alpha * u_orc||^2
    - scale_resid: mean residual after that projection (or epistemic if residual dies)

    Returns a dict consumed by uncertainty() / rollout_model().
    """
    ensemble.eval()
    n = min(num_samples, real_buffer.size)
    ind = np.random.randint(0, real_buffer.size, size=n)
    s = torch.FloatTensor(real_buffer.state[ind]).to(device)
    a = torch.FloatTensor(real_buffer.action[ind]).to(device)

    u_van = ensemble.uncertainty_vanilla(s, a).reshape(-1)
    u_orc = oracle_aleatoric_from_norm_batch(
        s,
        a,
        state_mean,
        state_std,
        beta=beta,
        reward_scale=reward_scale,
        n_actions=n_actions,
    ).reshape(-1)
    u_ep = ensemble.uncertainty_epistemic(s, a).reshape(-1)

    mean_van = float(max(u_van.mean().item(), 1e-6))
    mean_orc = float(max(u_orc.mean().item(), 1e-6))
    mean_ep = float(max(u_ep.mean().item(), 1e-6))

    # least-squares: u_van ≈ alpha * u_orc  (no intercept; both non-negative)
    denom = float((u_orc * u_orc).sum().item())
    numer = float((u_van * u_orc).sum().item())
    alpha = numer / max(denom, 1e-12)
    alpha = float(max(alpha, 0.0))

    resid = torch.relu(u_van - alpha * u_orc)
    mean_resid = float(resid.mean().item())

    # correlation diagnostic (how much pred-var is aleatoric)
    van_c = u_van - u_van.mean()
    orc_c = u_orc - u_orc.mean()
    corr = float(
        (van_c * orc_c).sum().item()
        / max(float(van_c.norm().item() * orc_c.norm().item()), 1e-12)
    )

    # If residual ≪ epistemic, pred-var is almost pure aleatoric — use epistemic as
    # the debiased penalty (oracle knows what remains after stripping ACK noise).
    use_ep_fallback = mean_resid < 0.05 * mean_ep or mean_resid < 1e-4
    if use_ep_fallback:
        scale_resid = mean_ep
        print(
            f"[penalty-norm] residual tiny (mean_resid={mean_resid:.3g} << mean_ep={mean_ep:.3g}); "
            f"oracle/debias falls back to epistemic"
        )
    else:
        scale_resid = max(mean_resid, 1e-6)

    print(
        f"[penalty-norm] mean_van={mean_van:.6g} mean_orc={mean_orc:.6g} mean_ep={mean_ep:.6g} "
        f"alpha={alpha:.4g} corr(van,orc)={corr:.3f} mean_resid={mean_resid:.6g} "
        f"scale_resid={scale_resid:.6g} ep_fallback={use_ep_fallback}"
    )
    return {
        "u_scale_van": mean_van,
        "u_scale_orc": mean_orc,
        "u_scale_resid": scale_resid,
        "alpha_orc": alpha,
        "use_ep_fallback": use_ep_fallback,
        "corr_van_orc": corr,
        "mean_resid": mean_resid,
        "mean_ep": mean_ep,
    }


def gaussian_nll(pred_mean, pred_logvar, target):
    inv_var = torch.exp(-pred_logvar)
    return 0.5 * (pred_logvar + (target - pred_mean) ** 2 * inv_var).mean()


def train_dynamics(ensemble, real_buffer, train_steps=DYNAMICS_TRAIN_STEPS, verbose: bool = False):
    optimizer = optim.Adam(ensemble.parameters(), lr=LR_DYNAMICS)
    ensemble.train()

    for step in range(1, train_steps + 1):
        s, a, r, next_s, _ = real_buffer.sample(BATCH_SIZE)
        delta_target = next_s - s
        target = torch.cat([delta_target, r], dim=-1)

        loss = 0.0
        for i in range(ensemble.num_models):
            mean, logvar = ensemble(s, a, model_idx=i)
            loss = loss + gaussian_nll(mean, logvar, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and step % 1000 == 0:
            print(f"  Dynamics step {step}/{train_steps}, loss: {loss.item():.4f}")

    ensemble.eval()


@torch.no_grad()
def rollout_model(
    ensemble,
    policy,
    init_states,
    max_action,
    penalty_coef,
    horizon,
    uncertainty_mode: str = "vanilla",
    state_mean: Optional[np.ndarray] = None,
    state_std: Optional[np.ndarray] = None,
    reward_scale: float = 1.0,
    beta: float = 0.5,
    n_actions: int = 28,
    u_scale_van: float = 1.0,
    u_scale_orc: float = 1.0,
    u_scale_resid: float = 1.0,
    alpha_orc: float = 1.0,
    use_ep_fallback: bool = False,
    return_diagnostics: bool = False,
):
    """Generate synthetic transitions in the learned model with uncertainty penalty.

    If return_diagnostics=True, also returns a dict with raw reward, u, and penalty.
    """
    states = init_states
    batch_size = states.shape[0]

    all_s, all_a, all_r, all_ns, all_done = [], [], [], [], []
    all_r_raw, all_u, all_pen = [], [], []

    for _ in range(horizon):
        actions_cont = policy.get_action_tensor(states)
        actions = continuous_to_discrete_torch(actions_cont, max_action)

        delta_s, reward, _ = ensemble.predict_one(states, actions, sample=True)
        u = ensemble.uncertainty(
            states,
            actions,
            mode=uncertainty_mode,
            state_mean=state_mean,
            state_std=state_std,
            reward_scale=reward_scale,
            beta=beta,
            n_actions=n_actions,
            u_scale_van=u_scale_van,
            u_scale_orc=u_scale_orc,
            u_scale_resid=u_scale_resid,
            alpha_orc=alpha_orc,
            use_ep_fallback=use_ep_fallback,
        )
        penalty = penalty_coef * u
        penalized_reward = reward - penalty
        next_states = states + delta_s

        all_s.append(states.cpu().numpy())
        all_a.append(actions.cpu().numpy())
        all_r.append(penalized_reward.cpu().numpy())
        all_ns.append(next_states.cpu().numpy())
        all_done.append(np.zeros((batch_size, 1), dtype=np.float32))
        if return_diagnostics:
            all_r_raw.append(reward.cpu().numpy())
            all_u.append(u.cpu().numpy())
            all_pen.append(penalty.cpu().numpy())

        states = next_states

    out = (
        np.concatenate(all_s, axis=0),
        np.concatenate(all_a, axis=0),
        np.concatenate(all_r, axis=0),
        np.concatenate(all_ns, axis=0),
        np.concatenate(all_done, axis=0),
    )
    if not return_diagnostics:
        return out
    diag = {
        "reward_raw": np.concatenate(all_r_raw, axis=0),
        "u": np.concatenate(all_u, axis=0),
        "penalty": np.concatenate(all_pen, axis=0),
    }
    return (*out, diag)


class SquashedGaussianActor(nn.Module):
    """Continuous actor on [0, max_action] (MCS index scale). Env uses round(a)."""

    def __init__(self, state_dim, action_dim, max_action, hidden=256, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.max_action = float(max_action)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def _squash(self, x_t):
        y = torch.tanh(x_t)
        action = (y + 1.0) * 0.5 * self.max_action
        log_det = torch.log(0.5 * self.max_action * (1.0 - y.pow(2)) + 1e-6)
        return action, log_det

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action, log_det = self._squash(x_t)
        log_prob = normal.log_prob(x_t) - log_det
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic(self, state):
        mean, _ = self.forward(state)
        action, _ = self._squash(mean)
        return action


class TwinQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)


class SACAgent:
    def __init__(self, state_dim, action_dim, max_action, n_actions: int):
        self.actor = SquashedGaussianActor(state_dim, action_dim, max_action).to(device)
        self.critic = TwinQNetwork(state_dim, action_dim).to(device)
        self.critic_target = TwinQNetwork(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LR_ALPHA)
        self.target_entropy = -float(action_dim)

        self.max_action = float(max_action)
        self.n_actions = int(n_actions)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def get_action_tensor(self, states):
        with torch.no_grad():
            return self.actor.deterministic(states)

    def get_action(self, s, state_mean, state_std):
        s_norm = (np.asarray(s, dtype=np.float32) - state_mean[0]) / state_std[0]
        s_t = torch.tensor(s_norm, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor.deterministic(s_t).squeeze(0).cpu().numpy()
        return np.clip(action, 0.0, self.max_action)

    def get_discrete_action(self, s, state_mean, state_std) -> int:
        a_cont = self.get_action(s, state_mean, state_std)
        return int(continuous_to_discrete(a_cont, self.n_actions)[0])

    def train_step(self, s, a, r, next_s, not_done):
        current_alpha = self.alpha.detach()

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_s)
            next_action = continuous_to_discrete_torch(next_action, self.max_action)
            target_q1, target_q2 = self.critic_target(next_s, next_action)
            target_q = torch.min(target_q1, target_q2) - current_alpha * next_log_prob
            target_value = r + GAMMA * not_done * target_q

        current_q1, current_q2 = self.critic(s, a)
        critic_loss = F.mse_loss(current_q1, target_value) + F.mse_loss(current_q2, target_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        new_action, log_prob = self.actor.sample(s)
        new_action_disc = continuous_to_discrete_torch(new_action, self.max_action)
        new_action_st = new_action + (new_action_disc - new_action).detach()
        q1, q2 = self.critic(s, new_action_st)
        q_pi = torch.min(q1, q2)
        actor_loss = (current_alpha * log_prob - q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self._soft_update(self.critic, self.critic_target)

    @staticmethod
    def _soft_update(net, target):
        for param, target_param in zip(net.parameters(), target.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)


def sample_mixed_batch(real_buffer, model_buffer, batch_size, real_ratio):
    real_batch = max(1, int(batch_size * real_ratio))
    model_batch = batch_size - real_batch

    real_samples = real_buffer.sample(real_batch)
    if model_buffer.size > 0 and model_batch > 0:
        model_samples = model_buffer.sample(model_batch)
        return tuple(torch.cat([r, m], dim=0) for r, m in zip(real_samples, model_samples))
    return real_samples


def evaluate(env, agent, state_mean, state_std, episodes=EVAL_EPISODES):
    avg_reward = 0.0
    for _ in range(episodes):
        s, _ = env.reset()
        done = False
        while not done:
            a = agent.get_discrete_action(s, state_mean, state_std)
            s, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            avg_reward += r
    return avg_reward / episodes


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MOPO on Appendix B LA (uncertainty ablations)")
    p.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))
    p.add_argument("--beta", type=float, default=None, help="override env beta; default from dataset meta")
    p.add_argument("--reward-scale", type=float, default=None, help="override reward_scale from meta")
    p.add_argument("--uncertainty-mode", type=str, default="vanilla", choices=list(UNCERTAINTY_MODES))
    p.add_argument("--dynamics-steps", type=int, default=DYNAMICS_TRAIN_STEPS)
    p.add_argument("--policy-steps", type=int, default=POLICY_TRAIN_STEPS)
    p.add_argument("--rollout-interval", type=int, default=ROLLOUT_INTERVAL)
    p.add_argument("--rollout-horizon", type=int, default=ROLLOUT_HORIZON)
    p.add_argument("--rollout-batch-size", type=int, default=ROLLOUT_BATCH_SIZE)
    p.add_argument("--penalty-coef", type=float, default=PENALTY_COEF)
    p.add_argument("--real-ratio", type=float, default=REAL_RATIO)
    p.add_argument("--eval-freq", type=int, default=EVAL_FREQ)
    p.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    p.add_argument("--eval-max-steps", type=int, default=EVAL_MAX_STEPS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-dir", type=str, default=str(ROOT / "results"))
    p.add_argument("--no-plot", action="store_true")
    return p


def train_mopo(args: argparse.Namespace) -> dict:
    """Run one MOPO training job; returns result dict and writes JSON."""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_path = Path(args.dataset)
    peek = np.load(dataset_path, allow_pickle=True)
    meta = json.loads(str(peek["meta_json"]))
    peek.close()

    beta = float(args.beta) if args.beta is not None else float(meta.get("beta", 0.5))
    reward_scale = (
        float(args.reward_scale)
        if args.reward_scale is not None
        else float(meta.get("reward_scale", 5.495217))
    )
    n_actions = int(meta.get("n_actions", 28))
    include_context = bool(meta.get("include_context", True))

    eval_env = make_la_env(
        beta=beta,
        reward_scale=reward_scale,
        n_actions=n_actions,
        n_states=int(meta.get("n_states", 5)),
        context_high=int(meta.get("context_high", 500)),
        include_context=include_context,
        max_episode_steps=args.eval_max_steps,
        seed=args.seed,
    )

    state_dim = int(eval_env.observation_space.shape[0])
    action_dim = 1
    max_action = float(n_actions - 1)

    real_buffer, meta = load_npz_buffer(dataset_path, state_dim, action_dim)
    state_mean, state_std = real_buffer.normalize_states()

    model_buffer = ReplayBuffer(state_dim, action_dim, max_size=max(real_buffer.size * 5, 100_000))
    model_buffer.copy_from(real_buffer)

    ensemble = EnsembleDynamics(state_dim, action_dim).to(device)
    train_dynamics(ensemble, real_buffer, train_steps=args.dynamics_steps)

    calib = calibrate_penalty_scales(
        ensemble,
        real_buffer,
        state_mean,
        state_std,
        beta=beta,
        reward_scale=reward_scale,
        n_actions=n_actions,
    )

    # Epistemic mode always normalizes by mean epistemic; residual path uses scale_resid.
    # When running epistemic as its own mode, point scale_resid at mean_ep.
    if args.uncertainty_mode == "epistemic":
        calib = {**calib, "u_scale_resid": calib["mean_ep"]}

    agent = SACAgent(state_dim, action_dim, max_action, n_actions=n_actions)

    eval_steps = []
    eval_rewards = []
    final_return = None

    for step in range(1, args.policy_steps + 1):
        if step % args.rollout_interval == 0:
            init_states = real_buffer.sample_states(args.rollout_batch_size)
            s, a, r, ns, dones = rollout_model(
                ensemble,
                agent,
                init_states,
                max_action,
                args.penalty_coef,
                args.rollout_horizon,
                uncertainty_mode=args.uncertainty_mode,
                state_mean=state_mean,
                state_std=state_std,
                reward_scale=reward_scale,
                beta=beta,
                n_actions=n_actions,
                u_scale_van=calib["u_scale_van"],
                u_scale_orc=calib["u_scale_orc"],
                u_scale_resid=calib["u_scale_resid"],
                alpha_orc=calib["alpha_orc"],
                use_ep_fallback=calib["use_ep_fallback"],
            )
            model_buffer.add_batch(s, a, r, ns, dones)

        batch = sample_mixed_batch(real_buffer, model_buffer, BATCH_SIZE, args.real_ratio)
        agent.train_step(*batch)

        if step % args.eval_freq == 0 or step == args.policy_steps:
            avg_reward = evaluate(
                eval_env, agent, state_mean, state_std, episodes=args.eval_episodes
            )
            eval_steps.append(step)
            eval_rewards.append(avg_reward)
            final_return = float(avg_reward)

    eval_env.close()

    result = {
        "method": f"mopo_{args.uncertainty_mode}",
        "uncertainty_mode": args.uncertainty_mode,
        "dataset": str(dataset_path),
        "epsilon": meta.get("epsilon"),
        "seed": args.seed,
        "beta": beta,
        "reward_scale": reward_scale,
        "penalty_coef": args.penalty_coef,
        "u_scale_van": calib["u_scale_van"],
        "u_scale_orc": calib["u_scale_orc"],
        "u_scale_resid": calib["u_scale_resid"],
        "alpha_orc": calib["alpha_orc"],
        "use_ep_fallback": calib["use_ep_fallback"],
        "corr_van_orc": calib["corr_van_orc"],
        "dynamics_steps": args.dynamics_steps,
        "policy_steps": args.policy_steps,
        "eval_max_steps": args.eval_max_steps,
        "eval_episodes": args.eval_episodes,
        "final_return": final_return,
        "eval_steps": eval_steps,
        "eval_rewards": eval_rewards,
        "dataset_mean_return": meta.get("mean_return"),
    }

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    eps_tag = meta.get("epsilon", "na")
    out_json = (
        results_dir
        / f"mopo_{args.uncertainty_mode}_eps{eps_tag}_seed{args.seed}_lam{args.penalty_coef:g}.json"
    )
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(
        f"[{args.uncertainty_mode}] eps={meta.get('epsilon')} "
        f"λ={args.penalty_coef:g} return={final_return:.2f} -> {out_json.name}"
    )

    if not args.no_plot and eval_steps:
        import matplotlib.pyplot as plt

        out_png = results_dir / f"mopo_{args.uncertainty_mode}_eps{eps_tag}_seed{args.seed}.png"
        plt.figure(figsize=(10, 5))
        plt.plot(eval_steps, eval_rewards, marker="o", linestyle="-", color="g")
        plt.title(f"MOPO ({args.uncertainty_mode}): {dataset_path.name}")
        plt.xlabel("Training Steps")
        plt.ylabel(f"Avg Return ({args.eval_max_steps} steps)")
        plt.grid(True)
        plt.savefig(out_png)
        plt.close()

    return result


def main(argv: Optional[list] = None):
    args = build_argparser().parse_args(argv)
    train_mopo(args)


if __name__ == "__main__":
    main()
