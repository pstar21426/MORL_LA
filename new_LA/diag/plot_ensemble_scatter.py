# Scatter: ensemble mean prediction vs real transition on the offline set.

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from train_mopo import EnsembleDynamics, apply_norm, load_dataset, one_hot

CKPT = ROOT / "outputs" / "mopo_slot_g090_seed0.pt"
DATA = ROOT / "datasets" / "offline_illa_olla.npz"
OUT = HERE / "outputs" / "ensemble_pred_scatter.png"


def main():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    n_models = 1 + max(
        int(k.split(".")[1]) for k in ckpt["ensemble"] if k.startswith("models.")
    )
    ensemble = EnsembleDynamics(
        int(ckpt["state_dim"]), int(ckpt["n_actions"]), num_models=n_models
    )
    ensemble.load_state_dict(ckpt["ensemble"])
    ensemble.eval()
    mean = ckpt["state_mean"]
    std = ckpt["state_std"]

    data, _ = load_dataset(DATA)
    rng = np.random.default_rng(0)
    n = len(data["rewards"])
    take = rng.choice(n, size=min(2500, n), replace=False)
    obs = data["observations"][take]
    nxt = data["next_observations"][take]
    act = data["actions"][take]
    rew = data["rewards"][take]

    s = torch.as_tensor(apply_norm(obs, mean, std), dtype=torch.float32)
    a = one_hot(torch.as_tensor(act), ensemble.n_actions)
    with torch.no_grad():
        means, logvars = ensemble(s, a)
    # means: (E, B, state + 2) = (delta in normalized state space, reward, slots / 13)
    ri = ensemble.reward_index
    delta = means[..., : ensemble.state_dim].mean(dim=0).numpy()
    pred_rew = means[..., ri].mean(dim=0).numpy()
    rew_var = torch.exp(logvars[..., ri]).mean(dim=0).numpy()
    pred_next = apply_norm(obs, mean, std) + delta
    pred_next = pred_next * std + mean

    q_i, cqi_i = 10, 0
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    panels = [
        (rew, pred_rew, "reward (bits/RE)", "true", "ensemble mean"),
        (nxt[:, q_i], pred_next[:, q_i], "next queue fraction", "true", "ensemble mean"),
        (nxt[:, cqi_i], pred_next[:, cqi_i], "next CQI (normalized)", "true", "ensemble mean"),
    ]
    for ax, (x, y, title, xl, yl) in zip(axes, panels):
        ax.scatter(x, y, s=6, c=rew_var, cmap="viridis", alpha=0.45, linewidths=0)
        lo = float(min(x.min(), y.min()))
        hi = float(max(x.max(), y.max()))
        ax.plot([lo, hi], [lo, hi], color="0.3", lw=1)
        ax.set_title(title)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(True, alpha=0.3)
    fig.colorbar(axes[0].collections[0], ax=axes, fraction=0.02, pad=0.02, label="mean reward variance")
    fig.suptitle("ensemble vs data (2500 transitions)", fontsize=11)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"saved {OUT}")
    print(
        "reward corr",
        round(float(np.corrcoef(rew, pred_rew)[0, 1]), 3),
        "queue corr",
        round(float(np.corrcoef(nxt[:, q_i], pred_next[:, q_i])[0, 1]), 3),
        "cqi corr",
        round(float(np.corrcoef(nxt[:, cqi_i], pred_next[:, cqi_i])[0, 1]), 3),
    )


if __name__ == "__main__":
    main()
