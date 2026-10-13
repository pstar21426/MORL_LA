"""Plot uncertainty diagnostics from train_uq.py NPZ output."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("npz", type=Path)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    data = np.load(args.npz)
    u_a = data["id_u_alea"]
    retx = data["id_retx"]
    u_e = data["id_u_epist"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    m0, m1 = u_a[retx == 0], u_a[retx > 0]
    ax.boxplot([m0, m1], tick_labels=["1st tx", "HARQ retx"])
    ax.set_title("Aleatoric entropy vs HARQ")
    ax.set_ylabel("u_alea")

    ax = axes[1]
    ax.scatter(1.0 / np.sqrt(np.arange(1, len(u_e) + 1)), u_e, s=6, alpha=0.35)
    ax.set_xlabel("1/sqrt(step index) proxy")
    ax.set_ylabel("u_epist")
    ax.set_title("Epistemic along eval rollout")

    out = args.out or args.npz.with_suffix(".png")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
