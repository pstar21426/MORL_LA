"""
Appendix A-style table: MOPO aleatoric overpessimism on toy LA.

Story
-----
1. Standard MOPO penalty absorbs irreducible ACK/NACK noise → over-penalizes.
2. Oracle aleatoric penalty (known p_success) isolates that overpessimism.
3. Oracle-debiased MOPO (strip aleatoric from u) recovers toward DP-greedy.

Modes (see algorithms/my_MOPO.py):
  vanilla   — predicted reward variance (aleatoric mixed in)
  oracle_p  — oracle Var[R|k,x,a] = p(1-p)(r_ok-r_fail)^2
  debias    — relu(vanilla - oracle_aleatoric)  ← expected ≈ greedy
  epistemic — ensemble mean disagreement only
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algorithms.my_MOPO import UNCERTAINTY_MODES, build_argparser as mopo_argparser, train_mopo
from data_collection.collect import collect_episode
from env.la_env import make_la_env
from utils.dp import solve_la_dp

EPSILONS = (0.0, 0.25, 0.5)

# Display names matching the paper-style claim
METHOD_LABELS = {
    "greedy": "Greedy (eps=0 DP)",
    "mopo_vanilla": "MOPO vanilla (pred var)",
    "mopo_oracle": "MOPO oracle/debias (-aleatoric)",
    "mopo_debias": "MOPO oracle/debias (-aleatoric)",
    "mopo_oracle_p": "MOPO oracle_p ablation (+aleatoric)",
    "mopo_epistemic": "MOPO epistemic only",
}


def dataset_path_for_eps(data_dir: Path, eps: float, num_steps: int = 10_000, seed: int = 0) -> Path:
    tag = f"eps{eps:g}_steps{num_steps}_ep1_seed{seed}"
    return data_dir / f"la_dp_{tag}.npz"


def eval_greedy(
    *,
    beta: float,
    reward_scale: float,
    num_steps: int,
    seed: int,
    epsilon: float = 0.0,
) -> float:
    env = make_la_env(
        beta=beta,
        reward_scale=reward_scale,
        max_episode_steps=num_steps,
        seed=seed,
    )
    _, policy = solve_la_dp(env, gamma=0.99)
    _, ep_ret = collect_episode(
        env,
        policy,
        epsilon=epsilon,
        seed=seed,
        rng=np.random.default_rng(seed + 10_000),
    )
    return float(ep_ret)


def run_greedy_grid(
    *,
    epsilons: Sequence[float],
    seeds: Sequence[int],
    beta: float,
    reward_scale: float,
    num_steps: int,
    results_dir: Path,
    greedy_epsilon: float = 0.0,
) -> List[Dict[str, Any]]:
    """Evaluate pure DP greedy (default ε=0) once per seed; tag rows with dataset ε."""
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        ret = eval_greedy(
            beta=beta,
            reward_scale=reward_scale,
            num_steps=num_steps,
            seed=seed,
            epsilon=float(greedy_epsilon),
        )
        for eps in epsilons:
            row = {
                "method": "greedy",
                "uncertainty_mode": None,
                "epsilon": float(eps),
                "seed": int(seed),
                "final_return": ret,
                "beta": beta,
                "reward_scale": reward_scale,
                "eval_max_steps": num_steps,
                "greedy_epsilon": float(greedy_epsilon),
            }
            out = results_dir / f"greedy_eps{eps:g}_seed{seed}.json"
            with open(out, "w", encoding="utf-8") as f:
                json.dump(row, f, indent=2)
            print(f"[greedy ε={greedy_epsilon:g}] tag_eps={eps:g} seed={seed} return={ret:.2f}")
            rows.append(row)
    return rows


def run_mopo_grid(
    *,
    modes: Sequence[str],
    epsilons: Sequence[float],
    seeds: Sequence[int],
    data_dir: Path,
    results_dir: Path,
    beta: float,
    reward_scale: float,
    dynamics_steps: int,
    policy_steps: int,
    eval_max_steps: int,
    eval_episodes: int,
    penalty_coefs: Sequence[float],
    rollout_batch_size: int,
    eval_freq: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    base = mopo_argparser().parse_args([])

    for mode in modes:
        for lam in penalty_coefs:
            for eps in epsilons:
                ds = dataset_path_for_eps(data_dir, eps)
                if not ds.exists():
                    raise FileNotFoundError(
                        f"missing dataset {ds}; run data_collection/collect.py first"
                    )
                for seed in seeds:
                    args = argparse.Namespace(**vars(base))
                    args.dataset = str(ds)
                    args.uncertainty_mode = mode
                    args.seed = int(seed)
                    args.beta = beta
                    args.reward_scale = reward_scale
                    args.dynamics_steps = dynamics_steps
                    args.policy_steps = policy_steps
                    args.eval_max_steps = eval_max_steps
                    args.eval_episodes = eval_episodes
                    args.penalty_coef = float(lam)
                    args.rollout_batch_size = rollout_batch_size
                    args.eval_freq = eval_freq
                    args.results_dir = str(results_dir)
                    args.no_plot = True
                    print(f"\n=== MOPO mode={mode} eps={eps:g} seed={seed} λ={lam:g} ===")
                    result = train_mopo(args)
                    rows.append(result)
    return rows


def aggregate_table(rows: List[Dict[str, Any]], epsilons: Sequence[float]) -> str:
    methods = []
    for r in rows:
        m = r["method"]
        if m not in methods:
            methods.append(m)

    lines = [
        "# MOPO aleatoric overpessimism (Appendix A toy LA)",
        "",
        "Expected pattern: vanilla collapses; oracle/debias survives (~ greedy).",
        "",
    ]
    header = "| Method | " + " | ".join(f"ε={e:g}" for e in epsilons) + " | Avg |"
    sep = "|--------|" + "|".join(["--------"] * len(epsilons)) + "|-----|"
    lines.append(header)
    lines.append(sep)

    for method in methods:
        cells = []
        means = []
        for eps in epsilons:
            vals = [
                float(r["final_return"])
                for r in rows
                if r["method"] == method and float(r["epsilon"]) == float(eps)
            ]
            if not vals:
                cells.append("—")
                continue
            mean = float(np.mean(vals))
            std = float(np.std(vals))
            means.append(mean)
            cells.append(f"{mean:.2f} ± {std:.2f}")
        avg = float(np.mean(means)) if means else float("nan")
        label = METHOD_LABELS.get(method, method)
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {avg:.2f} |")
    return "\n".join(lines)


def aggregate_lambda_table(rows: List[Dict[str, Any]], lambdas: Sequence[float]) -> str:
    """Pivot: rows = methods, columns = λ (greedy repeated across λ as ceiling)."""
    methods = []
    for r in rows:
        m = r["method"]
        if m not in methods:
            methods.append(m)

    lines = [
        "# MOPO lambda sweep (unit-mean penalty norm)",
        "",
        "Same λ is comparable across modes after normalizing u to mean 1 on real data.",
        "",
    ]
    header = "| Method | " + " | ".join(f"λ={lam:g}" for lam in lambdas) + " |"
    sep = "|--------|" + "|".join(["--------"] * len(lambdas)) + "|"
    lines.append(header)
    lines.append(sep)

    for method in methods:
        cells = []
        for lam in lambdas:
            if method == "greedy":
                vals = [float(r["final_return"]) for r in rows if r["method"] == "greedy"]
            else:
                vals = [
                    float(r["final_return"])
                    for r in rows
                    if r["method"] == method
                    and r.get("penalty_coef") is not None
                    and float(r["penalty_coef"]) == float(lam)
                ]
            if not vals:
                cells.append("-")
                continue
            mean = float(np.mean(vals))
            std = float(np.std(vals))
            cells.append(f"{mean:.2f} ± {std:.2f}")
        label = METHOD_LABELS.get(method, method)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fields = [
        "method",
        "uncertainty_mode",
        "epsilon",
        "seed",
        "final_return",
        "beta",
        "reward_scale",
        "eval_max_steps",
        "dataset",
        "penalty_coef",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MOPO table: Greedy vs vanilla vs oracle (chunkable / fast presets)"
    )
    p.add_argument("--data-dir", type=str, default=str(ROOT / "datasets"))
    p.add_argument("--results-dir", type=str, default=str(ROOT / "results"))
    # Only the three arms you care about (greedy is separate)
    p.add_argument("--modes", nargs="+", default=["vanilla", "oracle"])
    p.add_argument("--epsilons", type=float, nargs="+", default=[0.0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--reward-scale", type=float, default=5.495217)
    p.add_argument("--eval-max-steps", type=int, default=1000)
    p.add_argument("--eval-episodes", type=int, default=1)
    p.add_argument("--dynamics-steps", type=int, default=1000)
    p.add_argument("--policy-steps", type=int, default=3000)
    p.add_argument(
        "--penalty-coef",
        type=float,
        default=5.0,
        help="λ in r̃=r−λu (ignored if --penalty-coefs is set)",
    )
    p.add_argument(
        "--penalty-coefs",
        type=float,
        nargs="+",
        default=None,
        help="λ sweep, e.g. --penalty-coefs 1 5 10 20",
    )
    p.add_argument("--rollout-batch-size", type=int, default=1000)
    p.add_argument("--eval-freq", type=int, default=3000)
    p.add_argument(
        "--greedy-epsilon",
        type=float,
        default=0.0,
        help="DP greedy eval noise; keep 0 for true optimum ceiling",
    )
    p.add_argument("--skip-greedy", action="store_true")
    p.add_argument("--skip-mopo", action="store_true")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="~1 min pipeline check (tiny steps)",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="tens-of-minutes chunk: dyn=1k, policy=3k, eval=1k (default-ish)",
    )
    p.add_argument(
        "--full-paper",
        action="store_true",
        help="Appendix-scale: dyn=5k, policy=50k, eval=10k, eps={0,0.25,0.5}",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    dynamics_steps = args.dynamics_steps
    policy_steps = args.policy_steps
    eval_freq = args.eval_freq
    rollout_batch = args.rollout_batch_size
    eval_max_steps = args.eval_max_steps
    seeds = list(args.seeds)
    epsilons = list(args.epsilons)
    modes = list(args.modes)

    if args.smoke:
        dynamics_steps = 50
        policy_steps = 100
        eval_freq = 100
        rollout_batch = 128
        eval_max_steps = 200
        seeds = seeds[:1]
    elif args.full_paper:
        dynamics_steps = 5000
        policy_steps = 50000
        eval_freq = 10000
        rollout_batch = 5000
        eval_max_steps = 10000
        if epsilons == [0.25]:
            epsilons = [0.0, 0.25, 0.5]
    elif args.fast:
        # Explicit tens-of-minutes profile (also the new defaults)
        dynamics_steps = 1000
        policy_steps = 3000
        eval_freq = 3000
        rollout_batch = 1000
        eval_max_steps = 1000

    penalty_coefs = (
        list(args.penalty_coefs)
        if args.penalty_coefs is not None
        else [float(args.penalty_coef)]
    )

    print(
        f"Preset: modes={modes} eps={epsilons} seeds={seeds} λ={penalty_coefs} | "
        f"dyn={dynamics_steps} policy={policy_steps} eval={eval_max_steps}"
    )

    all_rows: List[Dict[str, Any]] = []

    if not args.skip_greedy:
        all_rows.extend(
            run_greedy_grid(
                epsilons=epsilons,
                seeds=seeds,
                beta=args.beta,
                reward_scale=args.reward_scale,
                num_steps=eval_max_steps,
                results_dir=results_dir,
                greedy_epsilon=args.greedy_epsilon,
            )
        )

    if not args.skip_mopo:
        all_rows.extend(
            run_mopo_grid(
                modes=modes,
                epsilons=epsilons,
                seeds=seeds,
                data_dir=data_dir,
                results_dir=results_dir,
                beta=args.beta,
                reward_scale=args.reward_scale,
                dynamics_steps=dynamics_steps,
                policy_steps=policy_steps,
                eval_max_steps=eval_max_steps,
                eval_episodes=args.eval_episodes,
                penalty_coefs=penalty_coefs,
                rollout_batch_size=rollout_batch,
                eval_freq=eval_freq,
            )
        )

    if len(penalty_coefs) > 1:
        table_md = aggregate_lambda_table(all_rows, penalty_coefs)
        print("\n=== Table (lambda sweep, unit-mean norm) ===")
    else:
        table_md = aggregate_table(all_rows, epsilons)
        print("\n=== Table (aleatoric overpessimism) ===")
    print(table_md)

    md_path = results_dir / "table_a1_mopo.md"
    md_path.write_text(table_md + "\n", encoding="utf-8")
    csv_path = results_dir / "table_a1_mopo.csv"
    write_csv(all_rows, csv_path)
    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
