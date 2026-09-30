# Seed 201, bias +3 dB, arrival 300-500. Same setup as
# illa_olla_bias3_arr300_seed201.png, extra slot-axis traces.

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sionna.sys import PHYAbstraction

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from la_env import DownlinkLAEnv, seed_phy
from policies import make_baseline_policy
from run_la_sim import _rolling, _slot_cum_return, rollout

SEED = 201
OUT = Path(__file__).resolve().parent / "outputs"


class _OffsetLog:
    def __init__(self, policy):
        self.policy = policy
        self.offset = []

    def reset(self):
        self.policy.reset()
        self.offset.clear()

    def __call__(self, state, info):
        action = self.policy(state, info)
        self.offset.append(float(getattr(self.policy, "_offset_db", np.nan)))
        return action


def load_config():
    with (ROOT / "configs" / "downlink_la.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _durations(num_slots):
    return np.maximum(np.asarray(num_slots, dtype=int), 1)


def _expand(values, num_slots, total, served):
    # Idle steps log num_slots=0 but still occupy one slot. Leave those as NaN.
    y = np.full(int(total), np.nan, dtype=np.float64)
    t = 0
    for v, ns, ok in zip(values, _durations(num_slots), served):
        end = min(t + int(ns), total)
        if ok and end > t:
            y[t:end] = float(v)
        t = end
        if t >= total:
            break
    return y


def _fail_on_served(ack, served):
    fail = np.full(len(ack), np.nan, dtype=np.float64)
    fail[served] = 1.0 - np.asarray(ack, dtype=np.float64)[served]
    return fail


def _running_bler(ack, served):
    fail = _fail_on_served(ack, served)
    out = np.full(len(ack), np.nan, dtype=np.float64)
    acc = 0.0
    n = 0
    last = np.nan
    for i, ok in enumerate(served):
        if not ok:
            out[i] = last
            continue
        n += 1
        acc += float(fail[i])
        last = acc / n
        out[i] = last
    return out


def _rolling_served(ack, served, window):
    fail = _fail_on_served(ack, served)
    out = np.full(len(ack), np.nan, dtype=np.float64)
    hist = []
    last = np.nan
    w = max(1, int(window))
    for i, ok in enumerate(served):
        if not ok:
            out[i] = last
            continue
        hist.append(float(fail[i]))
        if len(hist) > w:
            hist.pop(0)
        last = float(np.mean(hist))
        out[i] = last
    return out


def main():
    cfg = load_config()
    phy = PHYAbstraction()
    env = DownlinkLAEnv.from_config(cfg, phy_abs=phy)
    bler_target = float(cfg["bler_target"])
    n_re = float(env.num_allocated_re)
    total = int(env.num_slots)
    print(
        f"seed={SEED} bias={env.cqi_bias_db} dB "
        f"arrival={env.arrival_bits_min}-{env.arrival_bits_max} "
        f"gap={env.harq_retx_gap_slots} re={n_re:.0f}"
    )

    illa = make_baseline_policy("illa", env, bler_target=bler_target)
    olla = make_baseline_policy(
        "olla", env, bler_target=bler_target, olla_step_up_db=cfg.get("olla_step_up_db")
    )
    wrapped = {"ILLA": _OffsetLog(illa), "OLLA": _OffsetLog(olla)}
    results = {}
    for name, policy in wrapped.items():
        seed_phy(SEED)
        res = rollout(env, policy, SEED)
        res["offset_db"] = np.asarray(policy.offset, dtype=np.float64)
        results[name] = res
        served = res["num_slots"] > 0
        n = int(served.sum())
        discard_se = res["lost_se"] - res["n_overflow"] / n_re
        print(
            f"{name}: TBs {n} MCS {res['mcs_used'][served].mean():.1f} "
            f"1st-tx BLER {1.0 - res['ack'][served].mean():.3f} "
            f"slots/TB {res['num_slots'][served].mean():.2f} "
            f"retx {res['num_retx'][served].mean():.2f} "
            f"overflow {res['n_overflow'].sum() / n_re:.1f} bits/RE "
            f"discard {discard_se.sum():.1f} bits/RE "
            f"offset end {res['offset_db'][-1]:.2f} dB"
        )

    slots = np.arange(total)
    ref = results["OLLA"]
    fig, axes = plt.subplots(6, 1, figsize=(8, 12), sharex=True)
    title = "seed 201, bias +3 dB, arrival 300-500"
    colors = {"ILLA": "C0", "OLLA": "C1"}

    for name, res in results.items():
        served = res["num_slots"] > 0
        mcs = _expand(res["mcs_used"], res["num_slots"], total, served)
        filled = mcs.copy()
        last = next((v for v in filled if np.isfinite(v)), 0.0)
        for i, v in enumerate(filled):
            if np.isfinite(v):
                last = v
            else:
                filled[i] = last
        axes[0].step(slots, mcs, where="post", color=colors[name], alpha=0.35, lw=0.8)
        axes[0].plot(slots, _rolling(filled, 50), color=colors[name], lw=1.5, label=name)
    axes[0].set_ylabel("MCS")
    axes[0].set_title(title + "\nMCS, faint = each TB, line = 50-slot mean")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for name, res in results.items():
        served = res["num_slots"] > 0
        hold = np.ones(len(served), dtype=bool)
        bler = _expand(_running_bler(res["ack"], served), res["num_slots"], total, hold)
        recent = _expand(_rolling_served(res["ack"], served, 40), res["num_slots"], total, hold)
        axes[1].plot(slots, bler, color=colors[name], lw=1.6, label=f"{name} cumulative")
        axes[1].plot(slots, recent, color=colors[name], lw=1.0, alpha=0.75, label=f"{name} last 40 TB")
    axes[1].axhline(bler_target, color="k", ls="--", lw=0.8, label="target 0.1")
    axes[1].set_ylabel("first-tx BLER")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].legend(fontsize=8, ncol=2)
    axes[1].grid(True, alpha=0.3)

    for name, res in results.items():
        served = res["num_slots"] > 0
        dur = _expand(res["num_slots"], res["num_slots"], total, served)
        axes[2].step(slots, dur, where="post", color=colors[name], lw=0.8, label=name)
    axes[2].set_ylabel("slots / TB")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    off = _expand(
        ref["offset_db"],
        ref["num_slots"],
        total,
        np.ones(len(ref["num_slots"]), dtype=bool),
    )
    axes[3].plot(slots, off, color="C1", label="OLLA offset")
    axes[3].axhline(0.0, color="0.5", ls=":", lw=0.8)
    axes[3].axhline(-env.cqi_bias_db, color="0.3", ls="--", lw=0.8, label=f"-bias ({-env.cqi_bias_db:.0f} dB)")
    axes[3].set_ylabel("OLLA offset (dB)")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    for name, res in results.items():
        overflow_se = res["n_overflow"] / n_re
        discard_se = res["lost_se"] - overflow_se
        axes[4].plot(
            _slot_cum_return(overflow_se, res["num_slots"], total),
            color=colors[name],
            label=f"{name} overflow",
        )
        axes[4].plot(
            _slot_cum_return(discard_se, res["num_slots"], total),
            color=colors[name],
            ls="--",
            label=f"{name} HARQ discard",
        )
    axes[4].set_ylabel("cumulative bits/RE")
    axes[4].legend(fontsize=8, ncol=2)
    axes[4].grid(True, alpha=0.3)

    axes[5].plot(slots, ref["sinr_true_trace"][:total], color="0.35", lw=0.9, label="true SINR")
    axes[5].plot(slots, ref["sinr_hat_trace"][:total], color="C3", lw=0.8, alpha=0.85, label="CQI SINR")
    axes[5].set_ylabel("SINR (dB)")
    axes[5].set_xlabel("slot")
    axes[5].legend()
    axes[5].grid(True, alpha=0.3)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "illa_olla_bias3_arr300_seed201_traces.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
