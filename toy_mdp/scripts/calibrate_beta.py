"""Find reward_scale so DP-greedy 10k return ≈ paper ~12306 (beta fixed)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_collection.collect import collect_episode
from env.la_env import make_la_env
from utils.dp import solve_la_dp

TARGET = 12306.35


def greedy_return(beta: float, reward_scale: float, num_steps: int, seed: int = 0) -> float:
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
        epsilon=0.0,
        seed=seed,
        rng=np.random.default_rng(seed + 10_000),
    )
    return float(ep_ret)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--num-steps", type=int, default=10_000)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--scale", type=float, default=1.0, help="probe this scale; also suggest exact")
    args = p.parse_args()

    base = [greedy_return(args.beta, 1.0, args.num_steps, seed=s) for s in args.seeds]
    base_mean = float(np.mean(base))
    suggested = TARGET / max(base_mean, 1e-8)
    print(f"beta={args.beta} scale=1 mean={base_mean:.2f}")
    print(f"suggested_reward_scale={suggested:.6f} for target={TARGET}")

    scaled = [greedy_return(args.beta, suggested, args.num_steps, seed=s) for s in args.seeds]
    print(
        f"beta={args.beta} scale={suggested:.6f} "
        f"mean={float(np.mean(scaled)):.2f} ± {float(np.std(scaled)):.2f}"
    )

    if abs(args.scale - 1.0) > 1e-9 and abs(args.scale - suggested) > 1e-6:
        probe = [greedy_return(args.beta, args.scale, args.num_steps, seed=s) for s in args.seeds]
        print(
            f"beta={args.beta} scale={args.scale} "
            f"mean={float(np.mean(probe)):.2f} ± {float(np.std(probe)):.2f}"
        )


if __name__ == "__main__":
    main()
