# Offline MOPO on DownlinkLAEnv.
# Behavior data: ILLA and OLLA with epsilon-greedy. Both act from the observation only.
# Rollout reward is the predicted reward minus lambda times the disagreement
# of the ensemble mean rewards.
# gamma is a per-slot factor. One decision is one TB, so the backup uses
# gamma ** tau, where tau is the number of slots that TB occupied.

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sionna.sys import PHYAbstraction

from cqi import CQI_MAX
from ddqn import QNet
from la_env import DownlinkLAEnv, seed_phy
from policies import EpsilonGreedyPolicy, make_baseline_policy
from train_ddqn import held_out_eval_seeds

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def one_hot(actions, n_actions):
    return F.one_hot(actions.long().view(-1), n_actions).float()


SLOT_SCALE = 13.0


class EnsembleDynamics(nn.Module):
    """Each member predicts a Gaussian over (state delta, reward, slots/13)."""

    def __init__(self, state_dim, n_actions, num_models=5, hidden=256):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.num_models = num_models
        self.reward_index = state_dim
        out_dim = state_dim + 2
        self.models = nn.ModuleList(
            [self._net(state_dim + n_actions, hidden, out_dim * 2) for _ in range(num_models)]
        )

    @staticmethod
    def _net(in_dim, hidden, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, state, action_oh, model_idx=None):
        x = torch.cat([state, action_oh], dim=-1)
        idxs = range(self.num_models) if model_idx is None else (model_idx,)
        means, logvars = [], []
        for i in idxs:
            mean, logvar = torch.chunk(self.models[i](x), 2, dim=-1)
            means.append(mean)
            logvars.append(logvar.clamp(-10.0, 0.5))
        if model_idx is not None:
            return means[0], logvars[0]
        return torch.stack(means), torch.stack(logvars)

    def reward_disagreement(self, state, action_oh):
        # Std of the members' mean reward predictions. Ignores the Gaussian noise head.
        means, _ = self.forward(state, action_oh)
        i = self.reward_index
        return means[..., i : i + 1].std(dim=0)

    def predict(self, state, action_oh, model_idx):
        mean, logvar = self.forward(state, action_oh, model_idx=model_idx)
        pred = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        delta = pred[:, : self.state_dim]
        reward = pred[:, self.state_dim : self.state_dim + 1]
        # Duration uses the mean head so the discount is not extra noise.
        slots = (mean[:, self.state_dim + 1 : self.state_dim + 2] * SLOT_SCALE).clamp(1.0, SLOT_SCALE)
        return state + delta, reward, slots


def gaussian_nll(mean, logvar, target):
    inv_var = torch.exp(-logvar)
    return 0.5 * ((target - mean).pow(2) * inv_var + logvar).mean()


def collect_dataset(env, episodes_per, epsilon, seed0, bler_target, olla_step):
    n_actions = env.action_space.n
    obs, act, rew, nxt, done, trunc, slots = [], [], [], [], [], [], []
    names = ("illa", "olla")
    for p_i, name in enumerate(names):
        base = make_baseline_policy(
            name,
            env,
            bler_target=bler_target,
            olla_step_up_db=olla_step if name == "olla" else None,
        )
        for ep in range(episodes_per):
            seed = seed0 + p_i * episodes_per + ep
            policy = EpsilonGreedyPolicy(base, n_actions, epsilon=epsilon, seed=seed)
            policy.reset()
            seed_phy(seed)
            state, info = env.reset(seed=seed)
            finished = False
            n = 0
            while not finished:
                action = int(policy(state, info))
                next_state, reward, terminated, truncated, info = env.step(action)
                finished = bool(terminated or truncated)
                obs.append(np.asarray(state, dtype=np.float32))
                act.append(action)
                rew.append(float(reward))
                nxt.append(np.asarray(next_state, dtype=np.float32))
                # The time limit is not in the state, so a truncated step still bootstraps.
                done.append(float(terminated))
                trunc.append(float(truncated))
                slots.append(float(max(int(info["outcome"]["num_slots"]), 1)))
                state = next_state
                n += 1
            print(f"  collected {name} seed {seed}: {n} decisions")
    return {
        "observations": np.stack(obs).astype(np.float32),
        "actions": np.asarray(act, dtype=np.int64),
        "rewards": np.asarray(rew, dtype=np.float32),
        "next_observations": np.stack(nxt).astype(np.float32),
        "terminations": np.asarray(done, dtype=np.float32),
        "truncations": np.asarray(trunc, dtype=np.float32),
        "slots": np.asarray(slots, dtype=np.float32),
    }


def save_dataset(path, data, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, meta_json=json.dumps(meta), **data)
    print(f"Saved {len(data['rewards'])} transitions -> {path}")


def load_dataset(path):
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta_json"]))
    keys = ["observations", "actions", "rewards", "next_observations", "terminations"]
    keys += [k for k in ("truncations", "slots") if k in z.files]
    data = {k: z[k] for k in keys}
    return data, meta


def normalize_fit(obs):
    mean = obs.mean(axis=0, keepdims=True).astype(np.float32)
    std = obs.std(axis=0, keepdims=True).astype(np.float32) + 1e-3
    return mean, std


def apply_norm(x, mean, std):
    return (x - mean) / std


def _snap(v, steps):
    # History slots hold -1 until seen; otherwise the value is on a k / steps grid.
    return torch.where(v < -0.5, torch.full_like(v, -1.0), torch.round(v.clamp(0.0, 1.0) * steps) / steps)


class ObsProjector:
    """Puts sampled model states back on the observation grid of DownlinkLAEnv._state.

    Layout: [cqi, cqi lags x L, mcs lags x L, first-ACK lags x L, queue fraction].
    """

    def __init__(self, mean, std, num_lags, cqi_max, mcs_span):
        self.mean = torch.as_tensor(mean, dtype=torch.float32, device=DEVICE)
        self.std = torch.as_tensor(std, dtype=torch.float32, device=DEVICE)
        self.num_lags = int(num_lags)
        self.cqi_max = float(cqi_max)
        self.mcs_span = float(mcs_span)

    def __call__(self, x):
        L = self.num_lags
        raw = x * self.std + self.mean
        cqi = torch.round(raw[:, :1].clamp(0.0, 1.0) * self.cqi_max) / self.cqi_max
        cqi_lags = _snap(raw[:, 1 : 1 + L], self.cqi_max)
        mcs_lags = _snap(raw[:, 1 + L : 1 + 2 * L], self.mcs_span)
        ack = raw[:, 1 + 2 * L : 1 + 3 * L]
        ack = torch.where(ack < -0.5, torch.full_like(ack, -1.0), (ack >= 0.5).float())
        queue = raw[:, -1:].clamp(0.0, 1.0)
        out = torch.cat([cqi, cqi_lags, mcs_lags, ack, queue], dim=1)
        return (out - self.mean) / self.std


def train_dynamics(ensemble, states, actions, rewards, next_states, slots, steps, batch_size):
    opt = torch.optim.Adam(ensemble.parameters(), lr=1e-3)
    n = states.shape[0]
    idx_all = np.arange(n)
    ensemble.train()
    last = None
    for step in range(steps):
        loss = 0.0
        opt.zero_grad()
        for model in ensemble.models:
            take = np.random.choice(idx_all, batch_size, replace=True)
            s = states[take]
            a = one_hot(actions[take], ensemble.n_actions)
            target = torch.cat(
                [next_states[take] - s, rewards[take], slots[take] / SLOT_SCALE], dim=-1
            )
            mean, logvar = torch.chunk(model(torch.cat([s, a], dim=-1)), 2, dim=-1)
            logvar = logvar.clamp(-10.0, 0.5)
            loss = loss + gaussian_nll(mean, logvar, target)
        loss = loss / ensemble.num_models
        loss.backward()
        opt.step()
        last = float(loss.item())
        if (step + 1) % max(steps // 5, 1) == 0:
            print(f"  dynamics {step + 1}/{steps} nll={last:.4f}")
    ensemble.eval()
    return last


def penalty_scale(ensemble, states, actions, batch=4096):
    n = min(batch, states.shape[0])
    take = torch.randint(0, states.shape[0], (n,), device=states.device)
    with torch.no_grad():
        u = ensemble.reward_disagreement(states[take], one_hot(actions[take], ensemble.n_actions))
    scale = float(u.mean().clamp_min(1e-6).item())
    print(f"  mean-disagreement scale on data: {scale:.6g}")
    return scale


class DiscreteQ:
    def __init__(self, state_dim, n_actions, gamma, hidden=256, lr=1e-3, sync=200):
        self.n_actions = n_actions
        self.gamma = gamma
        self.sync = sync
        self.updates = 0
        self.q = QNet(state_dim, n_actions, hidden=hidden).to(DEVICE)
        self.q_target = QNet(state_dim, n_actions, hidden=hidden).to(DEVICE)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)

    def act(self, state_norm, greedy=False, epsilon=0.0):
        if not greedy and np.random.random() < epsilon:
            return int(np.random.randint(0, self.n_actions))
        s = torch.as_tensor(state_norm, dtype=torch.float32, device=DEVICE).view(1, -1)
        with torch.no_grad():
            return int(self.q(s).argmax(dim=1).item())

    def act_batch(self, states):
        with torch.no_grad():
            return self.q(states).argmax(dim=1)

    def update(self, s, a, r, ns, done, slots):
        q_sa = self.q(s).gather(1, a.long().view(-1, 1)).squeeze(1)
        with torch.no_grad():
            next_a = self.q(ns).argmax(dim=1)
            next_q = self.q_target(ns).gather(1, next_a.view(-1, 1)).squeeze(1)
            # gamma is per slot; one backup spans the slots this TB used.
            discount = self.gamma ** slots.view(-1).clamp(1.0, SLOT_SCALE)
            target = r.view(-1) + discount * next_q * (1.0 - done.view(-1))
        loss = F.mse_loss(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.updates += 1
        if self.updates % self.sync == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return float(loss.item())


def sample_rows(n, k):
    return torch.randint(0, n, (k,), device=DEVICE)


def mixed_batch(real, model, batch, real_ratio):
    n_real = int(round(batch * real_ratio)) if model["rewards"].shape[0] else batch
    n_real = min(max(n_real, 1), batch)
    n_model = batch - n_real
    parts = []
    for src, k in ((real, n_real), (model, n_model)):
        if k == 0:
            continue
        ix = sample_rows(src["states"].shape[0], k)
        parts.append(tuple(t[ix] for t in (
            src["states"], src["actions"], src["rewards"], src["next_states"], src["dones"], src["slots"]
        )))
    cols = zip(*parts)
    return tuple(torch.cat(c, dim=0) for c in cols)


@torch.no_grad()
def rollout(ensemble, agent, init_states, penalty_coef, u_scale, horizon, project):
    states = init_states
    ss, aa, rr, ns, dd, sl = [], [], [], [], [], []
    for _ in range(horizon):
        actions = agent.act_batch(states)
        oh = one_hot(actions, ensemble.n_actions)
        member = int(np.random.randint(0, ensemble.num_models))
        nxt, reward, pred_slots = ensemble.predict(states, oh, member)
        nxt = project(nxt)
        u = ensemble.reward_disagreement(states, oh) / u_scale
        penalized = reward - penalty_coef * u
        b = states.shape[0]
        ss.append(states)
        aa.append(actions.view(-1, 1).float())
        rr.append(penalized)
        ns.append(nxt)
        dd.append(torch.zeros(b, 1, device=DEVICE))
        sl.append(pred_slots)
        states = nxt
    return {
        "states": torch.cat(ss, 0),
        "actions": torch.cat(aa, 0),
        "rewards": torch.cat(rr, 0),
        "next_states": torch.cat(ns, 0),
        "dones": torch.cat(dd, 0),
        "slots": torch.cat(sl, 0),
    }


def evaluate(env, agent, mean, std, seed, bler_target):
    from train_ddqn import _rollout

    def choose(state, info):
        del info
        x = apply_norm(np.asarray(state, dtype=np.float32).reshape(1, -1), mean, std)
        return agent.act(x, greedy=True)

    seed_phy(seed)
    return _rollout(env, choose, seed)


def pack_tensors(data, mean, std):
    states = torch.as_tensor(apply_norm(data["observations"], mean, std), device=DEVICE)
    next_states = torch.as_tensor(
        apply_norm(data["next_observations"], mean, std), device=DEVICE
    )
    actions = torch.as_tensor(data["actions"], dtype=torch.float32, device=DEVICE).view(-1, 1)
    rewards = torch.as_tensor(data["rewards"], dtype=torch.float32, device=DEVICE).view(-1, 1)
    dones = torch.as_tensor(data["terminations"], dtype=torch.float32, device=DEVICE).view(-1, 1)
    slots = torch.as_tensor(data["slots"], dtype=torch.float32, device=DEVICE).view(-1, 1)
    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "next_states": next_states,
        "dones": dones,
        "slots": slots,
    }


def compare_heldout(env, agent, mean, std, seeds, bler_target, olla_step):
    from train_ddqn import _rollout

    agent.q.eval()
    illa = make_baseline_policy("illa", env, bler_target=bler_target)
    olla = make_baseline_policy(
        "olla", env, bler_target=bler_target, olla_step_up_db=olla_step
    )

    def mopo(state, info):
        del info
        x = apply_norm(np.asarray(state, dtype=np.float32).reshape(1, -1), mean, std)
        return agent.act(x, greedy=True)

    policies = (("ILLA", illa), ("OLLA", olla), ("MOPO", mopo))
    rows = {name: [] for name, _ in policies}
    for seed in seeds:
        for name, pol in policies:
            if hasattr(pol, "reset"):
                pol.reset()
            seed_phy(seed)
            metrics = _rollout(env, pol, seed)
            rows[name].append(metrics)
            print(
                f"  {name} seed {seed}: return={metrics['return']:.1f} "
                f"drops={metrics['drops']:.1f} BLER={metrics['first_tx_bler']:.3f} "
                f"MCS={metrics['mean_mcs']:.1f}"
            )
    print(f"Means over seeds {list(seeds)}")
    for name, _ in policies:
        ret = float(np.mean([m["return"] for m in rows[name]]))
        drops = float(np.mean([m["drops"] for m in rows[name]]))
        print(f"  {name}: return={ret:.1f} drops={drops:.1f}")


def main():
    p = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    p.add_argument("--config", type=Path, default=here / "configs" / "downlink_la.yaml")
    p.add_argument("--dataset", type=Path, default=here / "datasets" / "offline_illa_olla.npz")
    p.add_argument("--episodes-per-policy", type=int, default=10)
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--recollect", action="store_true")
    p.add_argument("--ensemble", type=int, default=5)
    p.add_argument("--dynamics-steps", type=int, default=2000)
    p.add_argument("--policy-steps", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--penalty-coef", type=float, default=1.0)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--rollout-batch", type=int, default=256)
    p.add_argument("--rollout-interval", type=int, default=250)
    p.add_argument("--real-ratio", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--num-slots", type=int, default=None)
    p.add_argument("--eval-seed", type=int, default=201)
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.num_slots is not None:
        cfg["num_slots"] = int(args.num_slots)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    phy = PHYAbstraction()
    env = DownlinkLAEnv.from_config(cfg, phy_abs=phy)
    state_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)
    bler = float(cfg["bler_target"])
    olla_step = cfg.get("olla_step_up_db")

    dataset_path = Path(args.dataset)
    if dataset_path.is_file() and not args.recollect:
        data, _meta = load_dataset(dataset_path)
        if "slots" not in data:
            args.recollect = True
            print("Dataset has no per-TB slot counts; recollecting.")
        elif "truncations" not in data:
            args.recollect = True
            print("Dataset marks time-limit truncation as terminal; recollecting.")
    if args.recollect or not dataset_path.is_file():
        print(
            f"Collect ILLA+OLLA eps={args.epsilon} "
            f"x {args.episodes_per_policy} episodes, slots={env.num_slots}"
        )
        data = collect_dataset(
            env,
            args.episodes_per_policy,
            args.epsilon,
            args.seed,
            bler,
            olla_step,
        )
        meta = {
            "epsilon": args.epsilon,
            "episodes_per_policy": args.episodes_per_policy,
            "seed0": args.seed,
            "policies": ["illa", "olla"],
            "num_slots": int(env.num_slots),
            "gamma_note": "slots is tau_k; the Q backup uses gamma ** tau_k",
        }
        save_dataset(dataset_path, data, meta)
    else:
        data, meta = load_dataset(dataset_path)
        print(f"Loaded {len(data['rewards'])} transitions <- {dataset_path}")

    tau, n_tau = np.unique(np.asarray(data["slots"]).astype(int), return_counts=True)
    print("tau counts", {int(a): int(b) for a, b in zip(tau, n_tau)})

    mean, std = normalize_fit(data["observations"])
    real = pack_tensors(data, mean, std)
    actions_long = real["actions"].view(-1).long()
    project = ObsProjector(mean, std, env.hist.num_lags, CQI_MAX, env._mcs_span)

    ensemble = EnsembleDynamics(state_dim, n_actions, num_models=args.ensemble).to(DEVICE)
    print(f"Train dynamics ensemble={args.ensemble} on {DEVICE}")
    train_dynamics(
        ensemble,
        real["states"],
        actions_long,
        real["rewards"],
        real["next_states"],
        real["slots"],
        args.dynamics_steps,
        args.batch_size,
    )
    u_scale = penalty_scale(ensemble, real["states"], actions_long)

    agent = DiscreteQ(state_dim, n_actions, gamma=args.gamma)
    model = {k: v[:0] for k, v in real.items()}
    empty_model = True
    print(
        f"Train Q steps={args.policy_steps} gamma={args.gamma} ** tau "
        f"penalty={args.penalty_coef} x ensemble-mean disagreement"
    )
    loss_sum = 0.0
    log_every = max(args.policy_steps // 10, 1)
    for step in range(1, args.policy_steps + 1):
        if step == 1 or step % args.rollout_interval == 0:
            take = sample_rows(real["states"].shape[0], args.rollout_batch)
            rolled = rollout(
                ensemble,
                agent,
                real["states"][take],
                args.penalty_coef,
                u_scale,
                args.horizon,
                project,
            )
            if empty_model:
                model = rolled
                empty_model = False
            else:
                model = {k: torch.cat([model[k], rolled[k]], 0) for k in model}
                cap = 200_000
                if model["rewards"].shape[0] > cap:
                    model = {k: v[-cap:] for k, v in model.items()}
        batch = mixed_batch(real, model, args.batch_size, args.real_ratio)
        loss = agent.update(*batch)
        loss_sum += loss
        if step % log_every == 0:
            print(
                f"  policy {step}/{args.policy_steps} "
                f"loss={loss_sum / log_every:.4f} model_n={model['rewards'].shape[0]}"
            )
            loss_sum = 0.0

    used = int(meta.get("episodes_per_policy", args.episodes_per_policy)) * 2
    eval_seeds = held_out_eval_seeds(
        int(meta.get("seed0", args.seed)), used, [201, 202, 203]
    )
    compare_heldout(env, agent, mean, std, eval_seeds, bler, olla_step)

    out = here_out(args.config, here)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.gamma:.2f}".replace(".", "")
    ckpt = out / f"mopo_slot_g{tag}_seed{args.seed}.pt"
    torch.save(
        {
            "q": agent.q.state_dict(),
            "state_dim": state_dim,
            "n_actions": n_actions,
            "state_mean": mean,
            "state_std": std,
            "penalty_coef": args.penalty_coef,
            "u_scale": u_scale,
            "gamma": args.gamma,
            "discount": "gamma ** tau",
            "ensemble": ensemble.state_dict(),
        },
        ckpt,
    )
    print(f"Saved -> {ckpt}")


def here_out(config_path, here):
    cfg = load_config(config_path)
    out = Path(cfg.get("out_dir", "outputs"))
    if not out.is_absolute():
        out = here / out
    return out


if __name__ == "__main__":
    main()
