"""
Appendix B toy LA용 Dynamic Programming.

완전 모델(전이·보상 식)이 있으므로 value iteration으로 Q*(k, x, a)를 구하고,
greedy policy π*(k, x) = argmax_a Q*(k, x, a)를 behavioral policy로 쓴다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from env.la_env import (
    CONTEXT_HIGH,
    DEFAULT_BETA,
    N_ACTIONS,
    N_STATES,
    SCALE_ACTION,
    LAEnv,
)


@dataclass
class DPSolution:
    """Value iteration 결과."""

    q: np.ndarray  # shape (n_states, context_high, n_actions), paper action 1..A → index 0..A-1
    v_bar: np.ndarray  # shape (n_states,): E_x[max_a Q(k, x, a)]
    greedy_actions: np.ndarray  # shape (n_states, context_high), paper action ∈ {1..n_actions}
    gamma: float
    beta: float
    n_iters: int
    max_delta: float


def _expected_backup(
    q: np.ndarray,
    v_bar: np.ndarray,
    *,
    gamma: float,
    beta: float,
    n_states: int,
    n_actions: int,
    context_high: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    한 번의 Bellman backup.

    Q(k, x, a) = p * (r_ok + γ V̄(0)) + (1-p) * (r_fail + γ V̄(k'))
      p = tanh(x / (18 * a))
      r_ok = tanh(a / n_actions)
      r_fail = -β (k + 1)
      k' = 0 if k == n_states-1 else k+1
    V̄(k) = mean_x max_a Q(k, x, a)
    """
    # paper actions a = 1..n_actions
    actions = np.arange(1, n_actions + 1, dtype=np.float64)  # (A,)
    contexts = np.arange(context_high, dtype=np.float64)  # (X,)

    # p[x, a_idx] = tanh(x / (18 * a))
    # x[:, None] / (18 * a[None, :]) → (X, A)
    p = np.tanh(contexts[:, None] / (SCALE_ACTION * actions[None, :]))  # (X, A)
    r_ok = np.tanh(actions / n_actions)  # (A,)
    # broadcast: (X, A)
    r_ok_mat = np.broadcast_to(r_ok, p.shape)

    q_new = np.empty_like(q)
    for k in range(n_states):
        r_fail = -beta * (k + 1)
        k_fail = 0 if k >= n_states - 1 else k + 1
        # Q(k, x, a) over all x, a
        q_new[k] = (
            p * (r_ok_mat + gamma * v_bar[0])
            + (1.0 - p) * (r_fail + gamma * v_bar[k_fail])
        )

    # V̄(k) = E_x[max_a Q(k,x,a)]
    v_bar_new = q_new.max(axis=2).mean(axis=1)
    return q_new, v_bar_new


def value_iteration(
    *,
    n_states: int = N_STATES,
    n_actions: int = N_ACTIONS,
    context_high: int = CONTEXT_HIGH,
    beta: float = DEFAULT_BETA,
    gamma: float = 0.99,
    tol: float = 1e-6,
    max_iters: int = 10_000,
    q_init: Optional[np.ndarray] = None,
) -> DPSolution:
    """
    Discounted value iteration → Q*, greedy policy.

    gamma=1.0은 continuing task에서 발산할 수 있어 기본은 0.99.
    평가 시 total return은 γ와 무관하게 undiscounted sum으로 재면 된다.
    """
    if not (0.0 <= gamma < 1.0):
        raise ValueError("gamma must be in [0, 1) for continuing LA MDP")
    if n_states < 1 or n_actions < 1 or context_high < 1:
        raise ValueError("n_states, n_actions, context_high must be >= 1")

    q = (
        np.zeros((n_states, context_high, n_actions), dtype=np.float64)
        if q_init is None
        else np.array(q_init, dtype=np.float64, copy=True)
    )
    if q.shape != (n_states, context_high, n_actions):
        raise ValueError(
            f"q_init shape {q.shape} != {(n_states, context_high, n_actions)}"
        )

    v_bar = q.max(axis=2).mean(axis=1)
    max_delta = float("inf")
    n_iters = 0

    for n_iters in range(1, max_iters + 1):
        q_new, v_bar_new = _expected_backup(
            q,
            v_bar,
            gamma=gamma,
            beta=beta,
            n_states=n_states,
            n_actions=n_actions,
            context_high=context_high,
        )
        max_delta = float(np.max(np.abs(q_new - q)))
        q = q_new
        v_bar = v_bar_new
        if max_delta < tol:
            break
    else:
        # max_iters 소진
        pass

    # greedy: paper action index (1..n_actions)
    greedy_idx = np.argmax(q, axis=2)  # 0-based action index
    greedy_actions = greedy_idx + 1

    return DPSolution(
        q=q,
        v_bar=v_bar,
        greedy_actions=greedy_actions,
        gamma=gamma,
        beta=beta,
        n_iters=n_iters,
        max_delta=max_delta,
    )


class GreedyPolicy:
    """
    DP로 구한 π*(k, x).

    select()는 paper action(1..28)을 반환하고,
    select_gym()은 Gym Discrete index(0..27)를 반환한다.
    """

    def __init__(self, solution: DPSolution) -> None:
        self.solution = solution
        self.greedy_actions = solution.greedy_actions  # (K, X)
        self.n_states, self.context_high = self.greedy_actions.shape
        self.n_actions = solution.q.shape[2]

    @classmethod
    def from_value_iteration(cls, **kwargs) -> "GreedyPolicy":
        return cls(value_iteration(**kwargs))

    def select(self, state: int, context: int) -> int:
        """Paper action ∈ {1..n_actions}."""
        if not (0 <= state < self.n_states):
            raise ValueError(f"state {state} out of range")
        if not (0 <= context < self.context_high):
            raise ValueError(f"context {context} out of range")
        return int(self.greedy_actions[state, context])

    def select_gym(self, state: int, context: int) -> int:
        """Gym Discrete action ∈ {0..n_actions-1}."""
        return self.select(state, context) - 1

    def q_values(self, state: int, context: int) -> np.ndarray:
        """Q*(state, context, ·) over paper actions, shape (n_actions,)."""
        return self.solution.q[state, context].copy()


def solve_la_dp(
    env: Optional[LAEnv] = None,
    *,
    gamma: float = 0.99,
    tol: float = 1e-6,
    max_iters: int = 10_000,
) -> Tuple[DPSolution, GreedyPolicy]:
    """
    LAEnv 설정에 맞춰 DP를 풀고 (solution, greedy policy)를 반환.

    env가 None이면 Appendix B 기본 하이퍼파라미터를 쓴다.
    """
    if env is None:
        kwargs = dict(
            n_states=N_STATES,
            n_actions=N_ACTIONS,
            context_high=CONTEXT_HIGH,
            beta=DEFAULT_BETA,
        )
    else:
        kwargs = dict(
            n_states=env.n_states,
            n_actions=env.n_actions,
            context_high=env.context_high,
            beta=env.beta,
        )

    sol = value_iteration(gamma=gamma, tol=tol, max_iters=max_iters, **kwargs)
    return sol, GreedyPolicy(sol)


def action_optimality_frequency(
    solution: DPSolution,
) -> np.ndarray:
    """
    Appendix B Fig.6용: 각 paper action이 몇 번 optimal인지 (state=0 기준 등).

    Returns
    -------
    freq : shape (n_states, n_actions)
        freq[k, a_idx] = #{x : greedy(k,x) == a_idx+1}
    """
    n_states, context_high, n_actions = solution.q.shape
    freq = np.zeros((n_states, n_actions), dtype=np.int64)
    for k in range(n_states):
        for a_paper in range(1, n_actions + 1):
            freq[k, a_paper - 1] = int(np.sum(solution.greedy_actions[k] == a_paper))
    return freq


if __name__ == "__main__":
    sol, pi = solve_la_dp(gamma=0.99)
    print(
        f"DP done: iters={sol.n_iters}, max_delta={sol.max_delta:.3e}, "
        f"gamma={sol.gamma}, beta={sol.beta}"
    )
    print(f"V_bar(k) = {np.array2string(sol.v_bar, precision=4)}")
    freq0 = action_optimality_frequency(sol)[0]
    top = np.argsort(freq0)[::-1][:5]
    print("Top-5 optimal actions at k=0 (paper index, count):")
    for a_idx in top:
        print(f"  a={a_idx + 1:2d}: {freq0[a_idx]}")
    # sanity: mid context should prefer mid-high MCS vs low context
    a_low_x = pi.select(0, 10)
    a_high_x = pi.select(0, 450)
    print(f"greedy(k=0, x=10)  = {a_low_x}")
    print(f"greedy(k=0, x=450) = {a_high_x}")
