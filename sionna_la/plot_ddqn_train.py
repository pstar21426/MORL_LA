import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


def load_config(path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def rolling_mean(x, window):
    x = np.asarray(x, dtype=np.float64)
    if window <= 1 or len(x) < window:
        return x.copy()
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def plot_ddqn_train(log, out_path, seed, roll_window=25):
    ep = log["episode"]
    ret = log["return"]
    tp = log["throughput"]
    bler = log["first_tx_bler"]
    eps = log["epsilon"]

    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # --- return ---
    axs[0].plot(ep, ret, color="C2", alpha=0.35, lw=0.8, label="episode return")
    if roll_window > 1 and len(ret) >= roll_window:
        rm = rolling_mean(ret, roll_window)
        axs[0].plot(
            ep[roll_window - 1 :],
            rm,
            color="C2",
            lw=2.0,
            label=f"{roll_window}-ep roll mean",
        )
    axs[0].set_ylabel("episode return")
    axs[0].set_ylim(bottom=min(0.0, float(np.min(ret)) * 1.05 if len(ret) else 0.0))
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc="best", fontsize=8)
    axs[0].set_title(f"DDQN training (seed={seed})")

    # --- throughput ---
    axs[1].plot(ep, tp, color="C0", alpha=0.35, lw=0.8, label="throughput")
    if roll_window > 1 and len(tp) >= roll_window:
        rm = rolling_mean(tp, roll_window)
        axs[1].plot(
            ep[roll_window - 1 :],
            rm,
            color="C0",
            lw=2.0,
            label=f"{roll_window}-ep roll mean",
        )
    axs[1].set_ylabel("throughput\n(return / num_slots)")
    axs[1].set_ylim(bottom=0)
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc="best", fontsize=8)

    # --- BLER + epsilon ---
    ax1 = axs[2]
    ax1.plot(ep, bler, color="C1", alpha=0.5, lw=0.8, label="1st-tx BLER")
    ax1.set_ylabel("1st-tx BLER")
    ax1.set_ylim(0, 1.0)
    ax1.set_xlabel("training episode")
    ax1.set_xlim(1, ep[-1] if len(ep) else 1)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(ep, eps, color="C3", lw=1.2, ls="--", label="epsilon")
    ax2.set_ylabel("epsilon")
    ax2.set_ylim(0, 1.05)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "downlink_la.yaml",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-npz", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--roll-window", type=int, default=8)
    args = p.parse_args()

    cfg = load_config(args.config)
    seed = args.seed
    out_dir = args.out_dir or Path(cfg.get("out_dir", "outputs"))
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parent / out_dir

    log_path = args.log_npz or out_dir / f"ddqn_train_log_seed{seed}.npz"
    if not log_path.is_file():
        raise FileNotFoundError(
            f"Training log not found: {log_path}\n"
            "Run train_ddqn.py first, e.g. python train_ddqn.py --episodes 1000"
        )

    log = {k: np.asarray(v) for k, v in np.load(log_path).items()}
    out_path = out_dir / f"ddqn_train_curve_seed{seed}.png"
    plot_ddqn_train(log, out_path, seed, roll_window=args.roll_window)

    print(f"Loaded {log_path}")
    print(f"  episodes: {len(log['episode'])}")
    print(f"  return: min={log['return'].min():.1f} max={log['return'].max():.1f}")
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
