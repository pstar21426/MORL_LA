# Offline Q on the real transitions only (no model), with and without a support mask.
# Support mask: per reported CQI, only MCS with >= MIN_COUNT samples are allowed in argmax.
# Also reports how often the saved MOPO greedy action falls outside that support.

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sionna.sys import PHYAbstraction

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from ddqn import QNet
from la_env import DownlinkLAEnv, seed_phy
from train_mopo import DEVICE, DiscreteQ, apply_norm, load_dataset, normalize_fit

SEEDS = (201, 202, 203, 204, 205)
MIN_COUNT = 10
STEPS = 30000


def support_mask(obs, act, n_actions):
    cqi = np.rint(obs[:, 0] * 15).astype(int)
    cnt = np.zeros((16, n_actions), dtype=np.int64)
    np.add.at(cnt, (cqi, act), 1)
    return cnt >= MIN_COUNT


def train_q(data, mean, std, gamma, mask, rate=None, per_decision=False):
    s = torch.as_tensor(apply_norm(data["observations"], mean, std), device=DEVICE)
    ns = torch.as_tensor(apply_norm(data["next_observations"], mean, std), device=DEVICE)
    a = torch.as_tensor(data["actions"], device=DEVICE).long()
    r = torch.as_tensor(data["rewards"], device=DEVICE)
    tau = torch.as_tensor(data["slots"], device=DEVICE)
    if rate is not None:
        r = r - rate * tau
    cq = np.rint(data["observations"][:, 0] * 15).astype(int)
    ncq = np.rint(data["next_observations"][:, 0] * 15).astype(int)
    m_next = None if mask is None else torch.as_tensor(mask[ncq], device=DEVICE)
    nA = 26
    q = QNet(s.shape[1], nA).to(DEVICE)
    qt = QNet(s.shape[1], nA).to(DEVICE)
    qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=1e-3)
    disc = gamma ** (torch.ones_like(tau) if per_decision else tau)
    n = s.shape[0]
    for it in range(STEPS):
        ix = torch.randint(0, n, (256,), device=DEVICE)
        with torch.no_grad():
            qn = q(ns[ix])
            if m_next is not None:
                qn = qn.masked_fill(~m_next[ix], -1e9)
            na = qn.argmax(1, keepdim=True)
            tgt = r[ix] + disc[ix] * qt(ns[ix]).gather(1, na).squeeze(1)
        loss = F.mse_loss(q(s[ix]).gather(1, a[ix].view(-1, 1)).squeeze(1), tgt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (it + 1) % 200 == 0:
            qt.load_state_dict(q.state_dict())
    q.eval()
    return q


def evaluate(env, q, mean, std, mask, track_support=None):
    rets, ood = [], []
    for seed in SEEDS:
        seed_phy(seed)
        state, info = env.reset(seed=seed)
        ret, done = 0.0, False
        while not done:
            x = torch.as_tensor(apply_norm(np.asarray(state, np.float32).reshape(1, -1), mean, std), device=DEVICE)
            with torch.no_grad():
                qv = q(x).squeeze(0)
            c = int(info["cqi_index"])
            if mask is not None:
                qv = qv.masked_fill(~torch.as_tensor(mask[c], device=DEVICE), -1e9)
            act = int(qv.argmax().item())
            if track_support is not None and info["queue"] > 0:
                ood.append(not track_support[c, act])
            state, rwd, term, trunc, info = env.step(act)
            ret += rwd
            done = term or trunc
        rets.append(ret)
    return float(np.mean(rets)), float(np.std(rets)), (float(np.mean(ood)) if ood else None)


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "downlink_la.yaml").open(encoding="utf-8"))
    env = DownlinkLAEnv.from_config(cfg, phy_abs=PHYAbstraction())
    data, _ = load_dataset(ROOT / "datasets" / "offline_illa_olla.npz")
    mean, std = normalize_fit(data["observations"])
    sup = support_mask(data["observations"], data["actions"], env.action_space.n)
    print(f"supported (CQI, MCS) cells with >= {MIN_COUNT} samples: {int(sup.sum())} / {sup.size}")

    ck = torch.load(ROOT / "outputs" / "mopo_slot_g090_seed0.pt", map_location="cpu", weights_only=False)
    mopo = DiscreteQ(int(ck["state_dim"]), int(ck["n_actions"]), gamma=0.9)
    mopo.q.load_state_dict(ck["q"])
    mopo.q.eval()
    r, sd, ood = evaluate(env, mopo.q, ck["state_mean"], ck["state_std"], None, track_support=sup)
    print(f"MOPO g0.90 saved           return={r:7.1f} ±{sd:5.1f}  greedy action unsupported: {ood:.2f}", flush=True)

    rate = 1.13  # OLLA bits/RE per slot on held-out seeds
    runs = [
        ("real-only g0.90^tau",              dict(gamma=0.90, mask=None)),
        ("real-only g0.90^tau + support",    dict(gamma=0.90, mask=sup)),
        ("real-only g0.97^tau + support",    dict(gamma=0.97, mask=sup)),
        ("per-decision g0.99 + support",     dict(gamma=0.99, mask=sup, per_decision=True)),
        ("r - rate*tau, g0.99/dec + support", dict(gamma=0.99, mask=sup, per_decision=True, rate=rate)),
    ]
    for name, kw in runs:
        torch.manual_seed(0)
        np.random.seed(0)
        q = train_q(data, mean, std, **kw)
        r, sd, ood = evaluate(env, q, mean, std, kw["mask"], track_support=sup)
        print(f"{name:34s} return={r:7.1f} ±{sd:5.1f}  greedy action unsupported: {ood:.2f}", flush=True)


if __name__ == "__main__":
    main()
