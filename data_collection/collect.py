"""
Appendix B: DP greedy(+ε)로 offline 데이터셋 수집.

메인 논문의 DQN behavioral policy 역할을, toy env에서는 DP greedy가 담당한다.
BCQ / CQL 등 value-based offline RL용 (s, a, r, s') 튜플을 저장한다.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from env.la_env import LAEnv, make_la_env
from utils.dp import GreedyPolicy, solve_la_dp


@dataclass
class CollectConfig:
    """데이터 수집 설정."""

    epsilon: float = 0.0
    num_steps: int = 10_000  # 에피소드 길이 (= env max_episode_steps)
    num_episodes: int = 1
    gamma_dp: float = 0.99  # DP value iteration discount
    beta: float = 0.5
    seed: int = 0
    include_context: bool = True
    out_dir: str = "datasets"
    tag: Optional[str] = None  # 파일명 접미사; None이면 eps/seed로 자동


class EpisodeBatch:
    """한 에피소드 transition 버퍼."""

    def __init__(self) -> None:
        self.observations: List[np.ndarray] = []
        self.actions: List[int] = []  # gym action 0..27
        self.rewards: List[float] = []
        self.next_observations: List[np.ndarray] = []
        self.terminations: List[bool] = []
        self.truncations: List[bool] = []
        self.states: List[int] = []  # step 시작 시점의 k
        self.contexts: List[int] = []  # step 시작 시점의 x
        self.paper_actions: List[int] = []
        self.successes: List[bool] = []

    def append(
        self,
        *,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool,
        state: int,
        context: int,
        paper_action: int,
        success: bool,
    ) -> None:
        self.observations.append(np.asarray(obs, dtype=np.float32))
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.next_observations.append(np.asarray(next_obs, dtype=np.float32))
        self.terminations.append(bool(terminated))
        self.truncations.append(bool(truncated))
        self.states.append(int(state))
        self.contexts.append(int(context))
        self.paper_actions.append(int(paper_action))
        self.successes.append(bool(success))

    def __len__(self) -> int:
        return len(self.rewards)


def select_action(
    policy: GreedyPolicy,
    *,
    state: int,
    context: int,
    epsilon: float,
    rng: np.random.Generator,
    n_actions: int,
) -> int:
    """ε-greedy → gym action (0-based)."""
    if epsilon > 0.0 and rng.random() < epsilon:
        return int(rng.integers(0, n_actions))
    return policy.select_gym(state, context)


def collect_episode(
    env: LAEnv,
    policy: GreedyPolicy,
    *,
    epsilon: float,
    seed: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[EpisodeBatch, float]:
    """
    한 에피소드 rollout.

    Returns
    -------
    batch : EpisodeBatch
    ep_return : undiscounted sum of rewards
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    obs, _info = env.reset(seed=seed)
    batch = EpisodeBatch()

    while True:
        # 의사결정 직전 내부 상태 (step 이후 info는 next state/context)
        state = int(env.state)
        context = int(env.context)

        action = select_action(
            policy,
            state=state,
            context=context,
            epsilon=epsilon,
            rng=rng,
            n_actions=env.n_actions,
        )

        next_obs, reward, terminated, truncated, next_info = env.step(action)
        paper_a = int(next_info["paper_action"])
        success = bool(next_info["success"])

        batch.append(
            obs=obs,
            action=action,
            reward=float(reward),
            next_obs=next_obs,
            terminated=terminated,
            truncated=truncated,
            state=state,
            context=context,
            paper_action=paper_a,
            success=success,
        )

        obs = next_obs
        if terminated or truncated:
            break

    ep_ret = float(np.sum(batch.rewards)) if len(batch) else 0.0
    return batch, ep_ret


def batches_to_arrays(batches: Sequence[EpisodeBatch]) -> Dict[str, np.ndarray]:
    """여러 에피소드를 concat한 flat transition dict (BCQ/CQL용)."""
    obs_list: List[np.ndarray] = []
    act_list: List[int] = []
    rew_list: List[float] = []
    next_obs_list: List[np.ndarray] = []
    term_list: List[bool] = []
    trunc_list: List[bool] = []
    state_list: List[int] = []
    ctx_list: List[int] = []
    paper_list: List[int] = []
    success_list: List[bool] = []
    ep_id_list: List[int] = []
    ep_returns: List[float] = []

    for ep_i, batch in enumerate(batches):
        ep_returns.append(float(np.sum(batch.rewards)) if len(batch) else 0.0)
        for t in range(len(batch)):
            obs_list.append(batch.observations[t])
            act_list.append(batch.actions[t])
            rew_list.append(batch.rewards[t])
            next_obs_list.append(batch.next_observations[t])
            term_list.append(batch.terminations[t])
            trunc_list.append(batch.truncations[t])
            state_list.append(batch.states[t])
            ctx_list.append(batch.contexts[t])
            paper_list.append(batch.paper_actions[t])
            success_list.append(batch.successes[t])
            ep_id_list.append(ep_i)

    return {
        "observations": np.stack(obs_list, axis=0).astype(np.float32),
        "actions": np.asarray(act_list, dtype=np.int64),
        "rewards": np.asarray(rew_list, dtype=np.float32),
        "next_observations": np.stack(next_obs_list, axis=0).astype(np.float32),
        "terminations": np.asarray(term_list, dtype=np.bool_),
        "truncations": np.asarray(trunc_list, dtype=np.bool_),
        "states": np.asarray(state_list, dtype=np.int64),
        "contexts": np.asarray(ctx_list, dtype=np.int64),
        "paper_actions": np.asarray(paper_list, dtype=np.int64),
        "successes": np.asarray(success_list, dtype=np.bool_),
        "episode_ids": np.asarray(ep_id_list, dtype=np.int64),
        "episode_returns": np.asarray(ep_returns, dtype=np.float32),
    }


def default_dataset_name(cfg: CollectConfig) -> str:
    tag = cfg.tag
    if tag is None:
        tag = f"eps{cfg.epsilon:g}_steps{cfg.num_steps}_ep{cfg.num_episodes}_seed{cfg.seed}"
    return f"la_dp_{tag}.npz"


def save_dataset(
    arrays: Dict[str, np.ndarray],
    meta: Dict[str, Any],
    path: Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, meta_json=np.asarray(json.dumps(meta)), **arrays)
    return path


def collect_dataset(
    cfg: CollectConfig,
    *,
    policy: Optional[GreedyPolicy] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], Path]:
    """
    DP greedy(+ε)로 데이터 수집 후 datasets/에 저장.

    Returns
    -------
    arrays, meta, save_path
    """
    if not (0.0 <= cfg.epsilon <= 1.0):
        raise ValueError("epsilon must be in [0, 1]")

    env = make_la_env(
        beta=cfg.beta,
        max_episode_steps=cfg.num_steps,
        include_context=cfg.include_context,
        seed=cfg.seed,
    )

    if policy is None:
        _, policy = solve_la_dp(env, gamma=cfg.gamma_dp)

    batches: List[EpisodeBatch] = []
    returns: List[float] = []

    for ep in range(cfg.num_episodes):
        ep_seed = int(cfg.seed + ep)
        batch, ep_ret = collect_episode(
            env,
            policy,
            epsilon=cfg.epsilon,
            seed=ep_seed,
            rng=np.random.default_rng(ep_seed + 10_000),
        )
        batches.append(batch)
        returns.append(ep_ret)

    arrays = batches_to_arrays(batches)
    meta = {
        "epsilon": cfg.epsilon,
        "num_steps": cfg.num_steps,
        "num_episodes": cfg.num_episodes,
        "gamma_dp": cfg.gamma_dp,
        "beta": cfg.beta,
        "seed": cfg.seed,
        "include_context": cfg.include_context,
        "n_states": env.n_states,
        "n_actions": env.n_actions,
        "context_high": env.context_high,
        "num_transitions": int(arrays["rewards"].shape[0]),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "behavioral_policy": "dp_greedy_epsilon",
    }

    out_path = Path(cfg.out_dir) / default_dataset_name(cfg)
    save_dataset(arrays, meta, out_path)
    return arrays, meta, out_path


def collect_epsilon_grid(
    epsilons: Sequence[float] = (0.0, 0.25, 0.5),
    *,
    num_steps: int = 10_000,
    num_episodes: int = 1,
    seed: int = 0,
    beta: float = 0.5,
    gamma_dp: float = 0.99,
    out_dir: str = "datasets",
) -> List[Dict[str, Any]]:
    """
    ε ∈ {0, 0.25, 0.5} 데이터셋을 한 번에 수집.
    DP는 한 번만 풀고 policy를 재사용한다.
    """
    env = make_la_env(beta=beta, max_episode_steps=num_steps, seed=seed)
    _, policy = solve_la_dp(env, gamma=gamma_dp)

    results: List[Dict[str, Any]] = []
    for eps in epsilons:
        cfg = CollectConfig(
            epsilon=float(eps),
            num_steps=num_steps,
            num_episodes=num_episodes,
            seed=seed,
            beta=beta,
            gamma_dp=gamma_dp,
            out_dir=out_dir,
        )
        _, meta, path = collect_dataset(cfg, policy=policy)
        results.append({"path": str(path), **meta})
        print(
            f"[eps={eps:g}] transitions={meta['num_transitions']} "
            f"mean_return={meta['mean_return']:.2f} -> {path}"
        )
    return results


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Collect Appendix B LA offline datasets")
    p.add_argument("--epsilon", type=float, default=None, help="single ε; omit to run grid")
    p.add_argument("--epsilons", type=float, nargs="+", default=[0.0, 0.25, 0.5])
    p.add_argument("--num-steps", type=int, default=10_000)
    p.add_argument("--num-episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--gamma-dp", type=float, default=0.99)
    p.add_argument("--out-dir", type=str, default="datasets")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_argparser().parse_args(argv)

    if args.epsilon is not None:
        cfg = CollectConfig(
            epsilon=args.epsilon,
            num_steps=args.num_steps,
            num_episodes=args.num_episodes,
            seed=args.seed,
            beta=args.beta,
            gamma_dp=args.gamma_dp,
            out_dir=args.out_dir,
        )
        _, meta, path = collect_dataset(cfg)
        print(
            f"[eps={args.epsilon:g}] transitions={meta['num_transitions']} "
            f"mean_return={meta['mean_return']:.2f} -> {path}"
        )
    else:
        collect_epsilon_grid(
            epsilons=args.epsilons,
            num_steps=args.num_steps,
            num_episodes=args.num_episodes,
            seed=args.seed,
            beta=args.beta,
            gamma_dp=args.gamma_dp,
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()
