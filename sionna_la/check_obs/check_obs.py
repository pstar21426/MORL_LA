#!/usr/bin/env python3
"""
DownlinkLAEnv observation / state verification (all-in-one).

Checks:
  1. Invariants during rollout (current CQI, step continuity, env._state match)
  2. History reconstruction via ACK-delay queue simulation
  3. CSI delay: gamma_hat[t] ~ gamma_true[t-d] (+ noise)
  4. PHY: mean predicted TBLER vs empirical first-tx BLER
  5. Scatter plots (CQI, MCS/ACK/CQI history, ILLA/OLLA oracle actions)
  6. ACK-delay timing plot (when ack_delay_slots > 0)
  7. DQN (--dqn): greedy consistency + vs ILLA baseline on same seed

Plots:
  obs_csi_*     : slot-axis gamma_true vs gamma_hat + CQI@decision
  obs_csi_scatter_* : square delay-aligned gamma scatter
  obs_basic_*   : state invariants
  obs_reconstruct_* : queue replay vs logged state
  obs_illa_oracle_* : ILLA reference actions
  obs_timing_*  : ACK delay (when > 0)

Run from sionna_la/:
  python check_obs/check_obs.py
  python check_obs/check_obs.py --ack-delay 0 --num-slots 200
  python check_obs/check_obs.py --ack-delay 8 --policy olla
  python check_obs/check_obs.py --dqn --checkpoint outputs/dqn_seed0.pt
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sionna.sys import PHYAbstraction

# Parent package imports (run: cd sionna_la)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dqn import DQNAgent  # noqa: E402
from la_env import DownlinkLAEnv, seed_phy  # noqa: E402
from policies import IllaPolicy, make_baseline_policy  # noqa: E402

_UNSEEN = -1.0


class _DQNPolicy:
    """Thin wrapper so DQNAgent fits rollout_detailed."""

    name = "dqn"

    def __init__(self, agent: DQNAgent):
        self.agent = agent

    def reset(self):
        pass

    def __call__(self, state, info):
        return self.agent.select_action(state, greedy=True)


def load_dqn_checkpoint(path: Path, env: DownlinkLAEnv) -> DQNAgent:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state_dim = int(ckpt.get("state_dim", np.prod(env.state_space.shape)))
    n_actions = int(ckpt.get("n_actions", env.action_space.n))
    hidden = int(ckpt.get("hidden", 256))
    agent = DQNAgent(state_dim, n_actions, hidden=hidden)
    agent.q.load_state_dict(ckpt["q"])
    agent.q_target.load_state_dict(ckpt["q"])
    agent.total_steps = int(ckpt.get("epsilon_decay_steps", agent.epsilon_decay_steps))
    return agent


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class CheckReport:
    name: str
    ok: int = 0
    fail: int = 0
    notes: list[str] = field(default_factory=list)

    def record(self, passed: bool, detail: str = ""):
        if passed:
            self.ok += 1
        else:
            self.fail += 1
            if detail:
                self.notes.append(detail)

    @property
    def passed(self) -> bool:
        return self.fail == 0

    def merge(self, other: CheckReport):
        self.ok += other.ok
        self.fail += other.fail
        self.notes.extend(other.notes)


def _state_slices(L: int):
    """Return index slices for [cqi_t, past_cqi(L), past_m(L), past_b(L)]."""
    i_cqi_p = 1 + L
    i_m = i_cqi_p + L
    i_b = i_m + L
    return slice(0, 1), slice(1, i_cqi_p), slice(i_cqi_p, i_m), slice(i_m, i_b)


def _state_dim(L: int) -> int:
    return 1 + 3 * L


def rollout_detailed(env: DownlinkLAEnv, policy, seed: int) -> dict:
    """Rollout with fields needed for observation verification."""
    state, info = env.reset(seed=seed)
    policy.reset()

    L = env.hist.num_lags
    log = {
        "state": [],
        "next_state": [],
        "action": [],
        "cqi_index": [],
        "cqi_norm": [],
        "env_state": [],
        "decision_slot": [],
        "finish_slot": [],
        "mcs_used": [],
        "mcs_norm": [],
        "ack": [],
        "sinr_true_decision": [],
        "sinr_hat_decision": [],
        "tbler": [],
        "tbler_first": [],
        "num_retx": [],
        "ack_delay": env.ack_delay_slots,
        "state_num_lags": L,
        "mcs_min": env.mcs_min,
        "mcs_span": env._mcs_span,
    }

    done = False
    while not done:
        env_state = env._state(info["cqi_index"])
        action = policy(state, info)

        log["state"].append(state.copy())
        log["env_state"].append(env_state.copy())
        log["action"].append(action)
        log["cqi_index"].append(int(info["cqi_index"]))
        log["cqi_norm"].append(float(state[0]))
        log["decision_slot"].append(int(info["slot"]))

        next_state, _reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        out = info["outcome"]

        mcs_used = int(out["mcs_used"])
        mcs_norm = (mcs_used - env.mcs_min) / env._mcs_span

        log["next_state"].append(next_state.copy())
        log["finish_slot"].append(int(out["decision_slot"]) + int(out["num_slots"]))
        log["mcs_used"].append(mcs_used)
        log["mcs_norm"].append(mcs_norm)
        log["ack"].append(int(out["ack"]))
        log["sinr_true_decision"].append(float(out["sinr_true_db"]))
        log["sinr_hat_decision"].append(float(out["sinr_hat_db"]))
        log["tbler"].append(float(out["tbler"]))
        log["tbler_first"].append(float(out["tbler_first"]))
        log["num_retx"].append(int(out["num_retx"]))
        state = next_state

    log["sinr_true_db"] = env._sinr_true_db.copy()
    log["sinr_hat_db"] = env._sinr_fb_db.copy()
    log["cqi_delay_slots"] = int(env._delay_used)
    log["cqi_noise_std_db"] = float(env.cqi_noise_std_db)
    log["num_slots"] = int(env.num_slots)
    log["bler_target"] = float(env.cqi_bler_target)
    log["seed"] = seed

    for k, v in log.items():
        if k in (
            "ack_delay",
            "state_num_lags",
            "mcs_min",
            "mcs_span",
            "cqi_delay_slots",
            "cqi_noise_std_db",
            "num_slots",
            "bler_target",
            "seed",
        ):
            continue
        if k in ("sinr_true_db", "sinr_hat_db"):
            continue
        log[k] = np.asarray(v)
    return log


def check_invariants(log: dict, tol: float = 1e-5) -> CheckReport:
    """Live invariant checks from rollout arrays."""
    rep = CheckReport("invariants")
    s = log["state"]
    ns = log["next_state"]
    n = len(s)
    L = int(log["state_num_lags"])
    _, sl_cqi_p, sl_m, sl_b = _state_slices(L)

    for i in range(n):
        cqi_norm = log["cqi_index"][i] / 15.0
        rep.record(
            abs(s[i, 0] - cqi_norm) <= tol,
            f"step {i}: state[0]={s[i,0]:.4f} != cqi/15={cqi_norm:.4f}",
        )
        rep.record(
            np.allclose(s[i], log["env_state"][i], atol=tol, rtol=0),
            f"step {i}: returned state != env._state()",
        )

    if n > 1:
        rep.record(
            np.allclose(ns[:-1], s[1:], atol=tol, equal_nan=True),
            "next_state[t] != state[t+1] (continuity break)",
        )

    # Last next_state is terminal (no following decision); skip history push checks there.
    hist_end = n - 1 if n else 0
    if log["ack_delay"] == 0:
        for i in range(hist_end):
            rep.record(
                abs(ns[i, 1] - log["cqi_index"][i] / 15.0) <= tol
                if L > 1
                else True,
                f"delay=0 step {i}: next_state past_cqi != decision cqi",
            )
            rep.record(
                abs(ns[i, sl_m.start] - log["mcs_norm"][i]) <= tol,
                f"delay=0 step {i}: next_state mcs history != outcome mcs_norm",
            )
            rep.record(
                abs(ns[i, sl_b.start] - log["ack"][i]) <= tol,
                f"delay=0 step {i}: next_state ack history != outcome ack",
            )

    return rep


def _push_hist(hist_cqi, hist_m, hist_b, cqi_n, m_norm, ack, L):
    hist_cqi = [cqi_n] + hist_cqi[: L - 1]
    hist_m = [m_norm] + hist_m[: L - 1]
    hist_b = [float(ack)] + hist_b[: L - 1]
    return hist_cqi, hist_m, hist_b


def simulate_expected_states(log: dict) -> np.ndarray:
    """Rebuild decision-time states from TB outcomes + ACK-delay queue."""
    L = int(log["state_num_lags"])
    ack_delay = int(log["ack_delay"])
    n = len(log["state"])

    hist_cqi = [_UNSEEN] * L
    hist_m = [_UNSEEN] * L
    hist_b = [_UNSEEN] * L
    fb_queue: list[tuple[int, int, float, float]] = []

    expected = np.full((n, _state_dim(L)), _UNSEEN, dtype=np.float64)

    for i in range(n):
        slot = int(log["decision_slot"][i])
        fb_queue.sort(key=lambda x: x[0])
        while fb_queue and fb_queue[0][0] <= slot:
            _, ack, cqi_n, m_norm = fb_queue.pop(0)
            hist_cqi, hist_m, hist_b = _push_hist(
                hist_cqi, hist_m, hist_b, cqi_n, m_norm, ack, L
            )

        cqi_n = log["cqi_index"][i] / 15.0
        expected[i] = np.array(
            [cqi_n, *hist_cqi, *hist_m, *hist_b], dtype=np.float64
        )

        delivery = int(log["finish_slot"][i]) + ack_delay
        fb_queue.append(
            (delivery, int(log["ack"][i]), log["cqi_index"][i] / 15.0, log["mcs_norm"][i])
        )

    return expected


def check_reconstructed_states(log: dict, tol: float = 1e-5) -> CheckReport:
    rep = CheckReport("reconstructed_state")
    expected = simulate_expected_states(log)
    actual = log["state"].astype(np.float64)
    mask = ~np.isclose(expected, _UNSEEN, atol=tol)
    if not mask.any():
        rep.record(True)
        return rep

    diffs = np.abs(expected[mask] - actual[mask])
    bad = int(np.sum(diffs > tol))
    rep.record(bad == 0, f"{bad} state elements differ from queue reconstruction")
    if bad:
        idx = np.argwhere(np.abs(expected - actual) > tol)
        for row in idx[:5]:
            i, j = int(row[0]), int(row[1])
            rep.notes.append(
                f"  step {i} dim {j}: logged={actual[i,j]:.4f} "
                f"expected={expected[i,j]:.4f}"
            )
    return rep


def _aligned_gamma_hat_pairs(gamma: np.ndarray, hat: np.ndarray, delay: int):
    """gamma_hat[t] uses delayed gamma: hat[t] ~ gamma[t-delay] for t >= delay."""
    n = min(len(gamma), len(hat))
    delay = max(0, int(delay))
    if delay >= n:
        return np.array([]), np.array([])
    t_idx = np.arange(delay, n)
    return gamma[t_idx - delay], hat[t_idx]


def check_csi_delay(log: dict, min_corr: float = 0.85) -> CheckReport:
    """CSI chain: hat trace follows true with configured CQI delay + noise."""
    rep = CheckReport("csi_delay")
    gamma = np.asarray(log["sinr_true_db"], dtype=np.float64)
    hat = np.asarray(log["sinr_hat_db"], dtype=np.float64)
    delay = int(log["cqi_delay_slots"])
    noise_std = float(log.get("cqi_noise_std_db", 1.5))

    x, y = _aligned_gamma_hat_pairs(gamma, hat, delay)
    if x.size < 10:
        rep.record(False, "too few aligned CSI samples")
        return rep

    corr = float(np.corrcoef(x, y)[0, 1])
    rep.record(
        corr >= min_corr,
        f"corr(gamma_true[t-{delay}], gamma_hat[t])={corr:.3f} (min {min_corr})",
    )

    resid = y - x
    rep.record(
        abs(float(np.mean(resid))) <= max(2.0 * noise_std, 0.5),
        f"mean residual {np.mean(resid):.3f} dB (noise_std={noise_std})",
    )
    rep.record(
        float(np.std(resid)) <= max(2.5 * noise_std, 1.0),
        f"residual std {np.std(resid):.3f} dB",
    )

    if noise_std < 1e-6:
        rep.record(
            float(np.max(np.abs(resid))) < 1e-3,
            f"noise=0: max |hat-true|={np.max(np.abs(resid)):.2e}",
        )
    return rep


def check_phy_tbler(log: dict, tol: float = 0.12) -> CheckReport:
    """Mean predicted first-tx TBLER vs empirical first-tx BLER."""
    rep = CheckReport("phy_tbler")
    tbler = np.asarray(log["tbler_first"], dtype=np.float64)
    ack = np.asarray(log["ack"], dtype=np.float64)
    if tbler.size == 0:
        rep.record(False, "no TB outcomes logged")
        return rep
    emp = float(1.0 - ack.mean())
    pred = float(tbler.mean())
    rep.record(
        abs(emp - pred) <= tol,
        f"emp 1st-tx BLER={emp:.3f} vs mean predicted TBLER={pred:.3f} (tol {tol})",
    )
    return rep


def check_policy_oracle(
    log: dict, env: DownlinkLAEnv, policy_name: str, bler_target: float, olla_step_up_db
) -> CheckReport:
    rep = CheckReport(f"{policy_name}_oracle")
    n = len(log["state"])
    policy = make_baseline_policy(
        policy_name,
        env,
        bler_target=bler_target,
        olla_step_up_db=olla_step_up_db,
    )

    # Replay decisions with the same delayed HARQ feedback sequence as rollout
    state, info = env.reset(seed=int(log.get("seed", 0)))
    policy.reset()
    done = False
    t = 0
    while not done and t < n:
        expected = policy(state, info)
        rep.record(
            int(expected) == int(log["action"][t]),
            f"step {t}: oracle action {expected} != logged {log['action'][t]}",
        )
        state, _, terminated, truncated, info = env.step(int(log["action"][t]))
        done = terminated or truncated
        t += 1

    return rep


def check_greedy_consistency(log: dict, agent: DQNAgent) -> CheckReport:
    """Greedy eval: re-argmax Q(s) must match logged actions."""
    rep = CheckReport("dqn_greedy_consistency")
    for i, state in enumerate(log["state"]):
        expected = agent.select_action(state, greedy=True)
        rep.record(
            int(expected) == int(log["action"][i]),
            f"step {i}: Q-greedy {expected} != logged {log['action'][i]}",
        )
    return rep


def illa_actions_from_log(log: dict, env: DownlinkLAEnv) -> np.ndarray:
    illa = IllaPolicy(
        mcs_min=env.mcs_min,
        mcs_max=env.mcs_max,
        mcs_table_index=env.mcs_table_index,
    )
    lo, hi = env.mcs_min, env.mcs_max
    actions = []
    for i in range(len(log["state"])):
        info = {"cqi_index": int(log["cqi_index"][i]), "mcs_min": lo, "mcs_max": hi}
        actions.append(illa(log["state"][i], info))
    return np.asarray(actions, dtype=np.int64)


def summarize_dqn_vs_illa(log: dict, env: DownlinkLAEnv) -> dict:
    illa_a = illa_actions_from_log(log, env)
    dqn_a = log["action"].astype(np.int64)
    agree = int(np.sum(illa_a == dqn_a))
    n = len(dqn_a)
    return {
        "illa_agree": agree,
        "illa_agree_frac": agree / max(n, 1),
        "mean_abs_action_diff": float(np.mean(np.abs(illa_a - dqn_a))),
        "illa_actions": illa_a,
    }


def plot_dqn_extra(
    log: dict,
    agent: DQNAgent,
    illa_actions: np.ndarray,
    seed: int,
    out_dir: Path,
):
    """DQN-only plots: greedy replay + vs ILLA + state coverage."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ack_delay = int(log["ack_delay"])
    tag = f"dqn_seed{seed}_ackd{ack_delay}"
    s = log["state"]
    n = len(s)

    greedy_actions = np.asarray(
        [agent.select_action(st, greedy=True) for st in s], dtype=np.int64
    )

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.5))

    _diag_scatter(
        axs[0],
        greedy_actions,
        log["action"],
        "DQN greedy replay",
        "Q(s).argmax replay",
        "logged action",
    )
    _diag_scatter(
        axs[1],
        illa_actions,
        log["action"],
        "DQN vs ILLA (same states)",
        "ILLA action",
        "DQN action",
    )

    unseen_per_step = np.sum(np.isclose(s, _UNSEEN, atol=1e-5), axis=1)
    axs[2].hist(unseen_per_step, bins=np.arange(-0.5, s.shape[1] + 1.5), color="C0", alpha=0.8)
    axs[2].set_xlabel("# unseen (-1) dims per state")
    axs[2].set_ylabel("count")
    axs[2].set_title("State coverage (unseen history)")
    axs[2].grid(True, alpha=0.3)
    axs[2].text(
        0.98,
        0.98,
        f"TBs={n}\nmean unseen={unseen_per_step.mean():.2f}",
        transform=axs[2].transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )

    fig.suptitle(f"DQN verification — {tag}", fontsize=11)
    fig.tight_layout()
    path = out_dir / f"obs_dqn_{tag}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def check_illa_direct(log: dict, env: DownlinkLAEnv) -> CheckReport:
    """ILLA is stateless: check action from logged (state, cqi_index) only."""
    rep = CheckReport("illa_direct")
    illa = IllaPolicy(
        mcs_min=env.mcs_min,
        mcs_max=env.mcs_max,
        mcs_table_index=env.mcs_table_index,
    )
    lo, hi = env.mcs_min, env.mcs_max
    for i in range(len(log["state"])):
        info = {
            "cqi_index": int(log["cqi_index"][i]),
            "mcs_min": lo,
            "mcs_max": hi,
        }
        expected = illa(log["state"][i], info)
        rep.record(
            int(expected) == int(log["action"][i]),
            f"step {i}: ILLA {expected} != {log['action'][i]}",
        )
    return rep


def _diag_scatter(ax, x, y, title, xlabel, ylabel):
    ax.scatter(x, y, s=8, alpha=0.5)
    lo = min(float(np.min(x)), float(np.min(y)))
    hi = max(float(np.max(x)), float(np.max(y)))
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    err = np.max(np.abs(np.asarray(x) - np.asarray(y)))
    ax.text(0.02, 0.98, f"max |Δ|={err:.2e}", transform=ax.transAxes, va="top", fontsize=8)


def plot_csi_scatter(
    log: dict,
    policy_name: str,
    seed: int,
    out_dir: Path,
) -> Path:
    """Square delay-aligned scatter: gamma_hat[t] vs gamma_true[t-d]."""
    out_dir.mkdir(parents=True, exist_ok=True)
    gamma = np.asarray(log["sinr_true_db"], dtype=np.float64)
    hat = np.asarray(log["sinr_hat_db"], dtype=np.float64)
    delay = int(log["cqi_delay_slots"])
    ack_delay = int(log["ack_delay"])
    tag = f"{policy_name}_seed{seed}_ackd{ack_delay}"
    x_align, y_align = _aligned_gamma_hat_pairs(gamma, hat, delay)

    fig, ax = plt.subplots(figsize=(6, 6))
    if x_align.size:
        ax.scatter(x_align, y_align, s=8, alpha=0.4, color="C2")
        lo = float(min(x_align.min(), y_align.min()))
        hi = float(max(x_align.max(), y_align.max()))
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.7)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.text(
            0.02,
            0.98,
            f"n={x_align.size}\nmean resid={np.mean(y_align - x_align):.2f} dB",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(rf"$\gamma_\mathrm{{true}}[t-{delay}]$ [dB]")
    ax.set_ylabel(r"$\hat{\gamma}[t]$ [dB]")
    ax.set_title(f"Delay-aligned CSI scatter — {tag}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / f"obs_csi_scatter_{tag}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_csi_verification(
    log: dict,
    policy_name: str,
    seed: int,
    out_dir: Path,
) -> tuple[Path, Path]:
    """Slot-axis CSI traces + decision CQI. Scatter is a separate square file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    gamma = np.asarray(log["sinr_true_db"], dtype=np.float64)
    hat = np.asarray(log["sinr_hat_db"], dtype=np.float64)
    delay = int(log["cqi_delay_slots"])
    noise_std = float(log.get("cqi_noise_std_db", 1.5))
    ack_delay = int(log["ack_delay"])
    tag = f"{policy_name}_seed{seed}_ackd{ack_delay}"

    n_show = min(len(gamma), int(log.get("num_slots", len(gamma))))
    t = np.arange(n_show)
    x_align, y_align = _aligned_gamma_hat_pairs(gamma, hat, delay)
    corr = float(np.corrcoef(x_align, y_align)[0, 1]) if x_align.size > 1 else float("nan")

    fig, axs = plt.subplots(2, 1, figsize=(10, 7))

    axs[0].plot(t, gamma[:n_show], label=r"$\gamma_\mathrm{true}$", color="C0", lw=1.0)
    axs[0].plot(t, hat[:n_show], label=r"$\hat{\gamma}$ (CQI input)", color="C1", lw=1.0, alpha=0.9)
    if delay > 0:
        shifted = np.full(n_show, np.nan)
        shifted[delay:n_show] = gamma[: n_show - delay]
        axs[0].plot(
            t,
            shifted,
            ":",
            color="C0",
            alpha=0.55,
            label=rf"$\gamma_\mathrm{{true}}[t-{delay}]$",
        )
    axs[0].set_ylabel("SINR [dB]")
    axs[0].set_title(
        f"CSI traces (cqi_delay={delay}, noise={noise_std:g} dB, corr={corr:.3f})"
    )
    axs[0].legend(loc="upper right", fontsize=8)
    axs[0].grid(True, alpha=0.3)

    tb = np.arange(len(log["cqi_index"]))
    axs[1].step(tb, log["cqi_index"], where="mid", color="C2", label="reported CQI")
    axs[1].set_ylabel("CQI index")
    axs[1].set_ylim(-0.5, 15.5)
    ax2 = axs[1].twinx()
    ax2.plot(
        tb,
        log["sinr_hat_decision"],
        "o",
        ms=2.5,
        alpha=0.45,
        color="C1",
        label=r"$\hat{\gamma}$ @ decision",
    )
    ax2.set_ylabel(r"$\hat{\gamma}$ @ decision [dB]", color="C1")
    axs[1].set_title("Decision-step CQI vs SINR hat")
    axs[1].set_xlabel("TB / decision index")
    axs[1].grid(True, alpha=0.3)
    lines1, labels1 = axs[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axs[1].legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    fig.suptitle(f"CSI verification — {tag}", fontsize=11)
    fig.tight_layout()
    path = out_dir / f"obs_csi_{tag}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path, plot_csi_scatter(log, policy_name, seed, out_dir)


def plot_verification(
    log: dict,
    expected: np.ndarray,
    policy_name: str,
    seed: int,
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    L = int(log["state_num_lags"])
    _, sl_cqi_p, sl_m, sl_b = _state_slices(L)
    s = log["state"]
    ns = log["next_state"]
    n = len(s)
    ack_delay = int(log["ack_delay"])
    tag = f"{policy_name}_seed{seed}_ackd{ack_delay}"

    # --- Figure 1: CQI + continuity + delay=0 history ---
    fig1, axs = plt.subplots(2, 2, figsize=(9, 8))
    _diag_scatter(
        axs[0, 0],
        log["cqi_index"],
        s[:, 0] * 15.0,
        "A. Current CQI",
        "info[cqi_index]",
        "state[0] × 15",
    )

    if n > 1:
        flat_logged = ns[:-1].ravel()
        flat_next = s[1:].ravel()
        mask = (flat_logged > _UNSEEN + 0.5) | (flat_next > _UNSEEN + 0.5)
        _diag_scatter(
            axs[0, 1],
            flat_logged[mask],
            flat_next[mask],
            "B. Step continuity",
            "next_state[t]",
            "state[t+1]",
        )
    else:
        axs[0, 1].set_title("B. Step continuity (n/a)")

    if ack_delay == 0 and L > 1:
        _diag_scatter(
            axs[1, 0],
            log["mcs_norm"],
            ns[:, sl_m.start],
            "C. MCS history (delay=0)",
            "outcome mcs_norm",
            f"next_state[{sl_m.start}]",
        )
        _diag_scatter(
            axs[1, 1],
            log["ack"],
            ns[:, sl_b.start],
            "D. ACK history (delay=0)",
            "outcome first_ack",
            f"next_state[{sl_b.start}]",
        )
    else:
        axs[1, 0].text(
            0.5,
            0.5,
            f"delay={ack_delay}: use timing plot\n+ reconstruction scatter",
            ha="center",
            va="center",
            transform=axs[1, 0].transAxes,
        )
        axs[1, 0].set_axis_off()
        axs[1, 1].set_axis_off()

    fig1.suptitle(f"Observation checks — {tag}", fontsize=11)
    fig1.tight_layout()
    p1 = out_dir / f"obs_basic_{tag}.png"
    fig1.savefig(p1, dpi=150)
    plt.close(fig1)

    # --- Figure 2: full state reconstruction ---
    fig2, ax = plt.subplots(figsize=(6, 6))
    mask = ~np.isclose(expected, _UNSEEN, atol=1e-5)
    _diag_scatter(
        ax,
        expected[mask].ravel(),
        s[mask].ravel(),
        "Reconstructed vs logged state",
        "queue-sim expected",
        "logged state",
    )
    p2 = out_dir / f"obs_reconstruct_{tag}.png"
    fig2.tight_layout()
    fig2.savefig(p2, dpi=150)
    plt.close(fig2)

    # --- Figure 3: ILLA oracle (always useful as reference) ---
    illa = IllaPolicy(
        mcs_min=int(log["mcs_min"]),
        mcs_max=int(log["mcs_min"]) + int(log["mcs_span"]),
        mcs_table_index=1,
    )
    expected_actions = []
    lo = int(log["mcs_min"])
    hi = lo + int(log["mcs_span"])
    for i in range(n):
        info = {"cqi_index": int(log["cqi_index"][i]), "mcs_min": lo, "mcs_max": hi}
        expected_actions.append(illa(s[i], info))
    expected_actions = np.asarray(expected_actions)

    fig3, ax = plt.subplots(figsize=(6, 6))
    _diag_scatter(
        ax,
        expected_actions,
        log["action"],
        f"ILLA oracle vs logged action ({policy_name} rollout)",
        "ILLA(state, info)",
        "logged action",
    )
    p3 = out_dir / f"obs_illa_oracle_{tag}.png"
    fig3.tight_layout()
    fig3.savefig(p3, dpi=150)
    plt.close(fig3)

    # --- Figure 4: ACK-delay timing (when delay > 0) ---
    p4 = None
    if ack_delay > 0:
        fig4, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        tb = np.arange(n)
        axs[0].step(tb, log["cqi_index"], where="mid", color="C2")
        axs[0].set_ylabel("CQI index")
        axs[0].grid(True, alpha=0.3)
        axs[0].set_title("History vs TB index (ACK delay > 0)")

        axs[1].plot(tb, s[:, sl_m.start], ".-", ms=3, label=f"state[{sl_m.start}] past MCS")
        axs[1].plot(tb, log["mcs_norm"], "x", ms=4, alpha=0.5, label="outcome mcs_norm")
        axs[1].set_ylabel("MCS norm")
        axs[1].legend(fontsize=8)
        axs[1].grid(True, alpha=0.3)

        axs[2].plot(tb, s[:, sl_b.start], ".-", ms=3, label=f"state[{sl_b.start}] past ACK")
        axs[2].plot(tb, log["ack"], "x", ms=4, alpha=0.5, label="outcome ack")
        delivery = log["finish_slot"] + ack_delay
        axs[2].vlines(
            tb,
            0,
            1,
            colors="gray",
            alpha=0.08,
            linewidth=8,
        )
        for i in range(min(n, 400)):
            axs[2].axvline(
                i,
                color="C3" if i > 0 and delivery[i - 1] <= log["decision_slot"][i] else "none",
                alpha=0.03,
            )
        axs[2].set_ylabel("first-tx ACK")
        axs[2].set_xlabel("TB / decision index")
        axs[2].legend(fontsize=8)
        axs[2].grid(True, alpha=0.3)

        fig4.tight_layout()
        p4 = out_dir / f"obs_timing_{tag}.png"
        fig4.savefig(p4, dpi=150)
        plt.close(fig4)

    return p1, p2, p3, p4


def print_report(reports: list[CheckReport], plots: list[Path], log: dict, extra: dict | None = None):
    print("\n=== Observation verification ===")
    print(
        f"TBs={len(log['state'])} L={log['state_num_lags']} "
        f"ack_delay={log['ack_delay']} cqi_delay={log.get('cqi_delay_slots', '?')} "
        f"dims={log['state'].shape[1]}"
    )
    if extra:
        for k, v in extra.items():
            if isinstance(v, float):
                print(f"  {k}={v:.4f}")
            else:
                print(f"  {k}={v}")

    all_ok = True
    for rep in reports:
        status = "PASS" if rep.passed else "FAIL"
        print(f"  [{status}] {rep.name}: ok={rep.ok} fail={rep.fail}")
        for note in rep.notes[:8]:
            print(f"         {note}")
        if len(rep.notes) > 8:
            print(f"         ... +{len(rep.notes) - 8} more")
        all_ok = all_ok and rep.passed

    print("\nPlots:")
    for p in plots:
        if p is not None:
            print(f"  {p}")

    print("\n" + ("All checks passed." if all_ok else "Some checks FAILED."))
    return all_ok


def run_dqn_checks(
    cfg: dict,
    seed: int,
    checkpoint: Path,
    ack_delay: int | None,
    num_slots: int | None,
    out_dir: Path,
    save_plots: bool,
) -> bool:
    if num_slots is not None:
        cfg = {**cfg, "num_slots": num_slots}
    if ack_delay is not None:
        cfg = {**cfg, "ack_delay_slots": ack_delay}

    seed_phy(seed)
    phy_abs = PHYAbstraction()
    env = DownlinkLAEnv.from_config(cfg, phy_abs=phy_abs)
    agent = load_dqn_checkpoint(checkpoint, env)
    policy = _DQNPolicy(agent)

    log = rollout_detailed(env, policy, seed=seed)

    vs_illa = summarize_dqn_vs_illa(log, env)
    reports = [
        check_invariants(log),
        check_reconstructed_states(log),
        check_csi_delay(log),
        check_phy_tbler(log),
        check_greedy_consistency(log, agent),
    ]

    expected = simulate_expected_states(log)
    plots: list[Path | None] = []
    if save_plots:
        p0, p0s = plot_csi_verification(log, "dqn", seed, out_dir)
        p1, p2, p3, p4 = plot_verification(log, expected, "dqn", seed, out_dir)
        p5 = plot_dqn_extra(log, agent, vs_illa["illa_actions"], seed, out_dir)
        plots = [p0, p0s, p1, p2, p3, p4, p5]

    extra = {
        "dqn_vs_illa_agree": f"{vs_illa['illa_agree']}/{len(log['action'])}",
        "dqn_vs_illa_frac": vs_illa["illa_agree_frac"],
        "mean_abs_action_diff_vs_illa": vs_illa["mean_abs_action_diff"],
        "checkpoint": str(checkpoint),
    }
    return print_report(reports, plots, log, extra=extra)


def run_checks(
    cfg: dict,
    seed: int,
    ack_delay: int | None,
    policy_name: str,
    num_slots: int | None,
    out_dir: Path,
    save_plots: bool,
) -> bool:
    if num_slots is not None:
        cfg = {**cfg, "num_slots": num_slots}
    if ack_delay is not None:
        cfg = {**cfg, "ack_delay_slots": ack_delay}

    bler_target = float(cfg["bler_target"])
    olla_step = cfg.get("olla_step_up_db")
    ack_delay_val = int(cfg["ack_delay_slots"])

    seed_phy(seed)
    phy_abs = PHYAbstraction()
    env = DownlinkLAEnv.from_config(cfg, phy_abs=phy_abs)
    policy = make_baseline_policy(
        policy_name, env, bler_target=bler_target, olla_step_up_db=olla_step
    )

    log = rollout_detailed(env, policy, seed=seed)

    reports = [
        check_invariants(log),
        check_reconstructed_states(log),
        check_csi_delay(log),
        check_phy_tbler(log),
    ]
    if policy_name == "illa":
        reports.append(check_illa_direct(log, env))
    else:
        reports.append(
            check_policy_oracle(log, env, policy_name, bler_target, olla_step)
        )

    expected = simulate_expected_states(log)
    plots: list[Path | None] = []
    if save_plots:
        p0, p0s = plot_csi_verification(log, policy_name, seed, out_dir)
        p1, p2, p3, p4 = plot_verification(
            log, expected, policy_name, seed, out_dir
        )
        plots = [p0, p0s, p1, p2, p3, p4]

    return print_report(reports, plots, log)


def main():
    p = argparse.ArgumentParser(description="DownlinkLAEnv observation verification")
    p.add_argument(
        "--config",
        type=Path,
        default=_ROOT / "configs" / "downlink_la.yaml",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-slots", type=int, default=200, help="shorter default for quick check")
    p.add_argument(
        "--ack-delay",
        type=int,
        default=None,
        help="override ack_delay_slots (default: run 0 and yaml value)",
    )
    p.add_argument(
        "--policy",
        choices=("illa", "olla"),
        default="illa",
        help="rollout policy for logging (ILLA checks are always plotted)",
    )
    p.add_argument(
        "--dqn",
        action="store_true",
        help="verify trained DQN (greedy eval + same obs checks)",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="DQN .pt path (default: outputs/dqn_seed{seed}.pt)",
    )
    p.add_argument(
        "--both-delays",
        action="store_true",
        help="run ack_delay=0 then yaml ack_delay_slots",
    )
    p.add_argument("--no-plot", action="store_true")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    yaml_ack = int(cfg.get("ack_delay_slots", 0))

    if args.dqn:
        ckpt = args.checkpoint
        if ckpt is None:
            out_base = Path(cfg.get("out_dir", "outputs"))
            if not out_base.is_absolute():
                out_base = _ROOT / out_base
            ckpt = out_base / f"dqn_seed{args.seed}.pt"
        if not ckpt.is_file():
            print(f"Checkpoint not found: {ckpt}")
            sys.exit(1)

        delays = [args.ack_delay] if args.ack_delay is not None else [yaml_ack]
        all_ok = True
        for d in delays:
            print(f"\n{'=' * 60}\n>>> DQN ack_delay={d} seed={args.seed}\n{'=' * 60}")
            ok = run_dqn_checks(
                cfg,
                seed=args.seed,
                checkpoint=ckpt,
                ack_delay=d,
                num_slots=args.num_slots,
                out_dir=args.out_dir,
                save_plots=not args.no_plot,
            )
            all_ok = all_ok and ok
        sys.exit(0 if all_ok else 1)

    delays = [args.ack_delay] if args.ack_delay is not None else []
    if args.both_delays or (args.ack_delay is None and not delays):
        delays = [0, yaml_ack] if yaml_ack != 0 else [0]

    all_ok = True
    for d in delays:
        label = f"ack_delay={d}"
        print(f"\n{'=' * 60}\n>>> {label} policy={args.policy} seed={args.seed}\n{'=' * 60}")
        ok = run_checks(
            cfg,
            seed=args.seed,
            ack_delay=d,
            policy_name=args.policy,
            num_slots=args.num_slots,
            out_dir=args.out_dir,
            save_plots=not args.no_plot,
        )
        all_ok = all_ok and ok

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
