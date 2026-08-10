"""
Scatter DP offline vs MOPO model-rollout data.

Defaults match results/lambda_sweep_norm_eps0:
  modes=vanilla,epistemic  eps=0  seed=0  λ∈{1,5,10,20}
  fast: dyn=1000, policy=3000, rollout every 1000 × batch 1000 × horizon 1
  unit-mean penalty normalization (same as train_mopo)

Example:
  python -m scripts.plot_rollout_scatter
  python -m scripts.plot_rollout_scatter --penalty-coefs 20 --modes vanilla epistemic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.my_MOPO import (
    BATCH_SIZE,
    EnsembleDynamics,
    ReplayBuffer,
    SACAgent,
    calibrate_penalty_scales,
    device,
    load_npz_buffer,
    rollout_model,
    sample_mixed_batch,
    train_dynamics,
)
from env.la_env import make_la_env


def denormalize_states(
    states: np.ndarray, state_mean: np.ndarray, state_std: np.ndarray
) -> np.ndarray:
    mean = np.asarray(state_mean, dtype=np.float32).reshape(1, -1)
    std = np.asarray(state_std, dtype=np.float32).reshape(1, -1)
    return states * std + mean


def collect_one(
    *,
    dataset_path: Path,
    mode: str,
    lam: float,
    seed: int,
    dynamics_steps: int,
    policy_steps: int,
    rollout_interval: int,
    rollout_batch_size: int,
    rollout_horizon: int,
    real_ratio: float,
    beta: float,
    reward_scale: float,
) -> Dict[str, np.ndarray]:
    """Mirror train_mopo (fast lambda-sweep settings) and keep all rollouts."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    peek = np.load(dataset_path, allow_pickle=True)
    meta = json.loads(str(peek["meta_json"]))
    peek.close()

    beta = float(beta if beta is not None else meta.get("beta", 0.5))
    reward_scale = float(
        reward_scale if reward_scale is not None else meta.get("reward_scale", 5.495217)
    )
    n_actions = int(meta.get("n_actions", 28))

    eval_env = make_la_env(
        beta=beta,
        reward_scale=reward_scale,
        n_actions=n_actions,
        n_states=int(meta.get("n_states", 5)),
        context_high=int(meta.get("context_high", 500)),
        include_context=bool(meta.get("include_context", True)),
        max_episode_steps=1000,
        seed=seed,
    )
    state_dim = int(eval_env.observation_space.shape[0])
    action_dim = 1
    max_action = float(n_actions - 1)
    eval_env.close()

    real_buffer, meta = load_npz_buffer(dataset_path, state_dim, action_dim)
    state_mean, state_std = real_buffer.normalize_states()

    # DP / real (normalized then denorm for plotting)
    real_s_norm = real_buffer.state[: real_buffer.size].copy()
    real_a = real_buffer.action[: real_buffer.size].copy()
    real_r = real_buffer.reward[: real_buffer.size].copy()
    real_s = denormalize_states(real_s_norm, state_mean, state_std)

    model_buffer = ReplayBuffer(state_dim, action_dim, max_size=max(real_buffer.size * 5, 100_000))
    model_buffer.copy_from(real_buffer)

    ensemble = EnsembleDynamics(state_dim, action_dim).to(device)
    train_dynamics(ensemble, real_buffer, train_steps=dynamics_steps)

    calib = calibrate_penalty_scales(
        ensemble,
        real_buffer,
        state_mean,
        state_std,
        beta=beta,
        reward_scale=reward_scale,
        n_actions=n_actions,
    )
    if mode == "epistemic":
        calib = {**calib, "u_scale_resid": calib["mean_ep"]}

    agent = SACAgent(state_dim, action_dim, max_action, n_actions=n_actions)

    roll_s, roll_a, roll_r_pen = [], [], []
    roll_r_raw, roll_u, roll_pen, roll_step = [], [], [], []

    for step in range(1, policy_steps + 1):
        if step % rollout_interval == 0:
            init_states = real_buffer.sample_states(rollout_batch_size)
            s, a, r, ns, dones, diag = rollout_model(
                ensemble,
                agent,
                init_states,
                max_action,
                lam,
                rollout_horizon,
                uncertainty_mode=mode,
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
                return_diagnostics=True,
            )
            model_buffer.add_batch(s, a, r, ns, dones)
            roll_s.append(denormalize_states(s, state_mean, state_std))
            roll_a.append(a)
            roll_r_pen.append(r)
            roll_r_raw.append(diag["reward_raw"])
            roll_u.append(diag["u"])
            roll_pen.append(diag["penalty"])
            roll_step.append(np.full((s.shape[0], 1), step, dtype=np.int32))

        batch = sample_mixed_batch(real_buffer, model_buffer, BATCH_SIZE, real_ratio)
        agent.train_step(*batch)

    return {
        "real_k": real_s[:, 0:1],
        "real_x": real_s[:, 1:2],
        "real_a": real_a,
        "real_r": real_r,
        "roll_k": np.concatenate(roll_s, axis=0)[:, 0:1],
        "roll_x": np.concatenate(roll_s, axis=0)[:, 1:2],
        "roll_a": np.concatenate(roll_a, axis=0),
        "roll_r_pen": np.concatenate(roll_r_pen, axis=0),
        "roll_r_raw": np.concatenate(roll_r_raw, axis=0),
        "roll_u": np.concatenate(roll_u, axis=0),
        "roll_penalty": np.concatenate(roll_pen, axis=0),
        "roll_step": np.concatenate(roll_step, axis=0),
        "mode": np.array(mode),
        "lam": np.array(lam, dtype=np.float32),
        "epsilon": np.array(meta.get("epsilon", 0.0), dtype=np.float32),
        "seed": np.array(seed, dtype=np.int32),
        "n_real": np.array(real_s.shape[0], dtype=np.int32),
        "n_roll": np.array(sum(x.shape[0] for x in roll_a), dtype=np.int32),
    }


def _subsample(n: int, max_n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= max_n:
        return np.arange(n)
    return rng.choice(n, size=max_n, replace=False)


def plot_scatter_pair(
    data: Dict[str, np.ndarray],
    out_png: Path,
    *,
    max_points: int = 4000,
    seed: int = 0,
    min_rollout_step: int = 3000,
) -> None:
    """Two panels: (x, a) and (k, x); DP vs rollout."""
    rng = np.random.default_rng(seed)
    n_real = int(data["real_a"].shape[0])
    roll_mask = data["roll_step"].ravel() >= int(min_rollout_step)
    roll_k = data["roll_k"][roll_mask]
    roll_x = data["roll_x"][roll_mask]
    roll_a = data["roll_a"][roll_mask]
    n_roll = int(roll_a.shape[0])
    ir = _subsample(n_real, max_points, rng)
    io = _subsample(n_roll, max_points, rng)

    mode = str(data["mode"])
    lam = float(data["lam"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(
        data["real_x"][ir, 0],
        data["real_a"][ir, 0],
        s=8,
        alpha=0.25,
        c="#4C72B0",
        label=f"DP real (n={n_real})",
        rasterized=True,
    )
    ax.scatter(
        roll_x[io, 0],
        roll_a[io, 0],
        s=10,
        alpha=0.35,
        c="#DD8452",
        label=f"rollout step>={min_rollout_step} (n={n_roll})",
        rasterized=True,
    )
    ax.set_xlabel("context x")
    ax.set_ylabel("action a (gym index 0..27)")
    ax.set_title(f"{mode}  λ={lam:g}  |  (x, a)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(
        data["real_k"][ir, 0],
        data["real_x"][ir, 0],
        s=8,
        alpha=0.25,
        c="#4C72B0",
        label="DP real",
        rasterized=True,
    )
    ax.scatter(
        roll_k[io, 0],
        roll_x[io, 0],
        s=10,
        alpha=0.35,
        c="#DD8452",
        label=f"rollout step>={min_rollout_step}",
        rasterized=True,
    )
    ax.set_xlabel("state k")
    ax.set_ylabel("context x")
    ax.set_title(f"{mode}  λ={lam:g}  |  (k, x)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "DP offline vs MOPO rollout (same settings as lambda_sweep_norm_eps0)",
        fontsize=11,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_penalty_colored(
    data: Dict[str, np.ndarray],
    out_png: Path,
    *,
    max_points: int = 4000,
    seed: int = 0,
    min_rollout_step: int = 3000,
) -> None:
    """Rollout-only (x, a) colored by penalty magnitude."""
    rng = np.random.default_rng(seed)
    roll_mask = data["roll_step"].ravel() >= int(min_rollout_step)
    roll_x = data["roll_x"][roll_mask]
    roll_a = data["roll_a"][roll_mask]
    roll_pen = data["roll_penalty"][roll_mask]
    n_roll = int(roll_a.shape[0])
    io = _subsample(n_roll, max_points, rng)
    mode = str(data["mode"])
    lam = float(data["lam"])
    pen = roll_pen[io, 0]

    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(
        roll_x[io, 0],
        roll_a[io, 0],
        c=pen,
        s=12,
        alpha=0.5,
        cmap="viridis",
        rasterized=True,
    )
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("λ · u  (penalty)")
    ax.set_xlabel("context x")
    ax.set_ylabel("action a (gym index)")
    ax.set_title(f"rollout penalty (step>={min_rollout_step})  |  {mode}  λ={lam:g}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def _filter_rollout(
    data: Dict[str, np.ndarray], min_rollout_step: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = data["roll_step"].ravel() >= int(min_rollout_step)
    return data["roll_k"][mask], data["roll_x"][mask], data["roll_a"][mask]


def plot_combined_triple(
    real_data: Dict[str, np.ndarray],
    vanilla_data: Dict[str, np.ndarray],
    epistemic_data: Dict[str, np.ndarray],
    out_png: Path,
    *,
    lam: float,
    max_points: int = 4000,
    seed: int = 0,
    min_rollout_step: int = 3000,
) -> None:
    """Blue=DP real, green=vanilla, orange=epistemic on the same axes."""
    rng = np.random.default_rng(seed)
    n_real = int(real_data["real_a"].shape[0])
    ir = _subsample(n_real, max_points, rng)

    vk, vx, va = _filter_rollout(vanilla_data, min_rollout_step)
    ek, ex, ea = _filter_rollout(epistemic_data, min_rollout_step)
    iv = _subsample(va.shape[0], max_points, rng)
    ie = _subsample(ea.shape[0], max_points, rng)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(
        real_data["real_x"][ir, 0],
        real_data["real_a"][ir, 0],
        s=8,
        alpha=0.22,
        c="#4C72B0",
        label=f"DP real (n={n_real})",
        rasterized=True,
        zorder=1,
    )
    ax.scatter(
        vx[iv, 0],
        va[iv, 0],
        s=12,
        alpha=0.4,
        c="#55A868",
        label=f"vanilla (n={va.shape[0]})",
        rasterized=True,
        zorder=2,
    )
    ax.scatter(
        ex[ie, 0],
        ea[ie, 0],
        s=12,
        alpha=0.4,
        c="#DD8452",
        label=f"epistemic (n={ea.shape[0]})",
        rasterized=True,
        zorder=3,
    )
    ax.set_xlabel("context x")
    ax.set_ylabel("action a (gym index 0..27)")
    ax.set_title(f"λ={lam:g}  |  (x, a)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(
        real_data["real_k"][ir, 0],
        real_data["real_x"][ir, 0],
        s=8,
        alpha=0.22,
        c="#4C72B0",
        label="DP real",
        rasterized=True,
        zorder=1,
    )
    ax.scatter(
        vk[iv, 0],
        vx[iv, 0],
        s=12,
        alpha=0.4,
        c="#55A868",
        label="vanilla",
        rasterized=True,
        zorder=2,
    )
    ax.scatter(
        ek[ie, 0],
        ex[ie, 0],
        s=12,
        alpha=0.4,
        c="#DD8452",
        label="epistemic",
        rasterized=True,
        zorder=3,
    )
    ax.set_xlabel("state k")
    ax.set_ylabel("context x")
    ax.set_title(f"λ={lam:g}  |  (k, x)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"DP / vanilla / epistemic rollouts (step>={min_rollout_step})",
        fontsize=11,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Scatter DP vs MOPO rollouts (lambda-sweep twin)")
    p.add_argument(
        "--dataset",
        type=str,
        default=str(ROOT / "datasets" / "la_dp_eps0_steps10000_ep1_seed0.npz"),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "results" / "lambda_sweep_norm_eps0" / "scatter"),
    )
    p.add_argument("--modes", nargs="+", default=["vanilla", "epistemic"])
    p.add_argument("--penalty-coefs", type=float, nargs="+", default=[1.0, 5.0, 10.0, 20.0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dynamics-steps", type=int, default=1000)
    p.add_argument("--policy-steps", type=int, default=3000)
    p.add_argument("--rollout-interval", type=int, default=1000)
    p.add_argument("--rollout-batch-size", type=int, default=1000)
    p.add_argument("--rollout-horizon", type=int, default=1)
    p.add_argument("--real-ratio", type=float, default=0.05)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--reward-scale", type=float, default=5.495217)
    p.add_argument("--max-plot-points", type=int, default=4000)
    p.add_argument(
        "--min-rollout-step",
        type=int,
        default=3000,
        help="Keep rollouts from this policy step onward (default 3000 = final batch only)",
    )
    p.add_argument(
        "--skip-train",
        action="store_true",
        help="Only re-plot from existing NPZs in out-dir",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.dataset)

    print(
        f"Scatter twin of lambda_sweep_norm_eps0 | modes={args.modes} "
        f"λ={args.penalty_coefs} dyn={args.dynamics_steps} policy={args.policy_steps}"
    )

    loaded: Dict[Tuple[str, float], Dict[str, np.ndarray]] = {}

    for mode in args.modes:
        for lam in args.penalty_coefs:
            tag = f"{mode}_lam{lam:g}_seed{args.seed}"
            npz_path = out_dir / f"rollouts_{tag}.npz"
            if args.skip_train:
                if not npz_path.exists():
                    raise FileNotFoundError(npz_path)
                data = dict(np.load(npz_path, allow_pickle=True))
                print(f"loaded {npz_path.name}  n_real={data['n_real']} n_roll={data['n_roll']}")
            else:
                print(f"\n=== collect mode={mode} λ={lam:g} ===")
                data = collect_one(
                    dataset_path=dataset_path,
                    mode=mode,
                    lam=float(lam),
                    seed=args.seed,
                    dynamics_steps=args.dynamics_steps,
                    policy_steps=args.policy_steps,
                    rollout_interval=args.rollout_interval,
                    rollout_batch_size=args.rollout_batch_size,
                    rollout_horizon=args.rollout_horizon,
                    real_ratio=args.real_ratio,
                    beta=args.beta,
                    reward_scale=args.reward_scale,
                )
                np.savez_compressed(npz_path, **data)
                print(
                    f"saved {npz_path.name}  n_real={int(data['n_real'])} "
                    f"n_roll={int(data['n_roll'])}"
                )

            loaded[(mode, float(lam))] = data
            plot_scatter_pair(
                data,
                out_dir / f"scatter_{tag}.png",
                max_points=args.max_plot_points,
                seed=args.seed,
                min_rollout_step=args.min_rollout_step,
            )
            plot_penalty_colored(
                data,
                out_dir / f"penalty_{tag}.png",
                max_points=args.max_plot_points,
                seed=args.seed,
                min_rollout_step=args.min_rollout_step,
            )
            print(f"wrote scatter_{tag}.png  penalty_{tag}.png")

    # Combined: blue=DP, green=vanilla, orange=epistemic (per λ)
    for lam in args.penalty_coefs:
        key_v = ("vanilla", float(lam))
        key_e = ("epistemic", float(lam))
        if key_v not in loaded or key_e not in loaded:
            print(f"skip combined λ={lam:g} (need both vanilla and epistemic NPZs)")
            continue
        van = loaded[key_v]
        epi = loaded[key_e]
        out = out_dir / f"scatter_combined_lam{lam:g}_seed{args.seed}.png"
        plot_combined_triple(
            van,  # real fields identical across modes
            van,
            epi,
            out,
            lam=float(lam),
            max_points=args.max_plot_points,
            seed=args.seed,
            min_rollout_step=args.min_rollout_step,
        )
        print(f"wrote {out.name}")

    print(f"\nDone. Figures in {out_dir}")


if __name__ == "__main__":
    main()
