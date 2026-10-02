# No bootstrap. Fit E[r | s, a] and E[tau | s, a] on the offline set, then act
# argmax_a E[r] - rho * E[tau] over supported actions. rho is updated to the
# measured bits/RE per slot of the previous policy (Dinkelbach).

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sionna.sys import PHYAbstraction

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from la_env import DownlinkLAEnv, seed_phy
from train_mopo import DEVICE, apply_norm, load_dataset, normalize_fit

SEEDS = (201, 202, 203, 204, 205)
MIN_COUNT = 10


def fit(data, mean, std, steps=8000):
    s = torch.as_tensor(apply_norm(data["observations"], mean, std), device=DEVICE)
    a = F.one_hot(torch.as_tensor(data["actions"], device=DEVICE).long(), 26).float()
    y = torch.stack(
        [torch.as_tensor(data["rewards"], device=DEVICE), torch.as_tensor(data["slots"], device=DEVICE)], 1
    )
    net = nn.Sequential(nn.Linear(s.shape[1] + 26, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 2)).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(steps):
        ix = torch.randint(0, s.shape[0], (512,), device=DEVICE)
        loss = F.mse_loss(net(torch.cat([s[ix], a[ix]], 1)), y[ix])
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    return net


def evaluate(env, net, mean, std, sup, rho):
    eye = torch.eye(26, device=DEVICE)
    rets, slots_total = [], 0
    for seed in SEEDS:
        seed_phy(seed)
        state, info = env.reset(seed=seed)
        ret, done = 0.0, False
        while not done:
            x = torch.as_tensor(apply_norm(np.asarray(state, np.float32).reshape(1, -1), mean, std), device=DEVICE)
            with torch.no_grad():
                pr = net(torch.cat([x.expand(26, -1), eye], 1))
            score = pr[:, 0] - rho * pr[:, 1]
            score = score.masked_fill(~torch.as_tensor(sup[int(info["cqi_index"])], device=DEVICE), -1e9)
            state, r, term, trunc, info = env.step(int(score.argmax().item()))
            ret += r
            done = term or trunc
        rets.append(ret)
        slots_total += env.num_slots
    return float(np.mean(rets)), float(np.std(rets)), float(np.sum(rets) / slots_total)


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "downlink_la.yaml").open(encoding="utf-8"))
    env = DownlinkLAEnv.from_config(cfg, phy_abs=PHYAbstraction())
    data, _ = load_dataset(ROOT / "datasets" / "offline_illa_olla.npz")
    mean, std = normalize_fit(data["observations"])
    cqi = np.rint(data["observations"][:, 0] * 15).astype(int)
    cnt = np.zeros((16, 26), dtype=np.int64)
    np.add.at(cnt, (cqi, data["actions"]), 1)
    sup = cnt >= MIN_COUNT

    torch.manual_seed(0)
    net = fit(data, mean, std)
    rho = 0.0
    for it in range(4):
        r, sd, rate = evaluate(env, net, mean, std, sup, rho)
        print(f"iter {it}: rho={rho:.3f}  return={r:7.1f} ±{sd:5.1f}  measured rate={rate:.3f}", flush=True)
        rho = rate
    for rho in (1.0, 1.2, 1.4):
        r, sd, rate = evaluate(env, net, mean, std, sup, rho)
        print(f"fixed rho={rho:.2f}  return={r:7.1f} ±{sd:5.1f}", flush=True)


if __name__ == "__main__":
    main()
