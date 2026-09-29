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
from run_la_sim import _slot_cum_return, rollout

SEEDS = (201, 202, 203)
OUT = Path(__file__).resolve().parent / "outputs"


def load_config():
    with (ROOT / "configs" / "downlink_la.yaml").open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["arrival_bits_min"] = 300
    cfg["arrival_bits_max"] = 500
    return cfg


def main():
    cfg = load_config()
    phy = PHYAbstraction()
    env = DownlinkLAEnv.from_config(cfg, phy_abs=phy)
    bler_target = float(cfg["bler_target"])
    n_re = env.num_allocated_re
    print(
        f"arrival={env.arrival_bits_min}-{env.arrival_bits_max} "
        f"gap={env.harq_retx_gap_slots} rho={env.sinr_ar_rho} re={n_re}"
    )
    illa = make_baseline_policy("illa", env, bler_target=bler_target)
    olla = make_baseline_policy(
        "olla", env, bler_target=bler_target, olla_step_up_db=cfg.get("olla_step_up_db")
    )
    policies = (("ILLA", illa), ("OLLA", olla))
    rows = []
    seed201 = {}
    queues = {}
    orig_arrive = env._arrive

    for seed in SEEDS:
        for name, policy in policies:
            qlog = []

            def wrapped(qlog=qlog):
                overflow = orig_arrive()
                qlog.append((int(env._t), int(env._q)))
                return overflow

            env._arrive = wrapped
            seed_phy(seed)
            res = rollout(env, policy, seed)
            env._arrive = orig_arrive
            served = res["num_slots"] > 0
            delivered_se = float(res["delivered_se"].sum())
            n = int(served.sum())
            row = {
                "seed": seed,
                "name": name,
                "delivered_bits": delivered_se * n_re,
                "bits_per_slot": delivered_se * n_re / env.num_slots,
                "return": float(res["reward"].sum()),
                "bler": float(1.0 - res["ack"][served].mean()) if n else 0.0,
                "mcs": float(res["mcs_used"][served].mean()) if n else 0.0,
                "slots": float(res["num_slots"][served].mean()) if n else 0.0,
                "overflow_bits": int(res["n_overflow"].sum()),
                "final_q": int(env._q),
            }
            rows.append(row)
            print(
                f"seed {seed} {name}: delivered {row['delivered_bits']:.0f} bits "
                f"({row['bits_per_slot']:.1f}/slot) return {row['return']:.1f} "
                f"BLER {row['bler']:.3f} MCS {row['mcs']:.1f} "
                f"slots/TB {row['slots']:.2f} overflow {row['overflow_bits']} "
                f"final_q {row['final_q']}"
            )
            if seed == 201:
                slots_tb = res["num_slots"]
                seed201[name] = _slot_cum_return(
                    res["delivered_se"], slots_tb, env.num_slots
                )
                # delivered bits/RE minus tail drop and HARQ discard, unweighted.
                net_se = np.asarray(res["delivered_se"], dtype=np.float64) - np.asarray(
                    res["lost_se"], dtype=np.float64
                )
                seed201[name + "_net"] = _slot_cum_return(net_se, slots_tb, env.num_slots)
                queues[name] = qlog

    print("--- mean ---")
    for name, _ in policies:
        sub = [r for r in rows if r["name"] == name]
        print(
            f"{name}: delivered {np.mean([r['delivered_bits'] for r in sub]):.0f} bits "
            f"({np.mean([r['bits_per_slot'] for r in sub]):.1f}/slot) "
            f"BLER {np.mean([r['bler'] for r in sub]):.3f} "
            f"overflow {np.mean([r['overflow_bits'] for r in sub]):.0f} "
            f"final_q {np.mean([r['final_q'] for r in sub]):.0f}"
        )

    for name in ("ILLA", "OLLA"):
        print(
            f"seed 201 {name}: delivered {seed201[name][-1]:.1f} bits/RE, "
            f"delivered-drops {seed201[name + '_net'][-1]:.1f} bits/RE"
        )

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(seed201["ILLA"], label="ILLA", color="C0")
    axes[0].plot(seed201["OLLA"], label="OLLA", color="C1")
    axes[0].set_ylabel("delivered bits/RE")
    axes[0].set_title("seed 201, arrival 300-500 bits/slot")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(seed201["ILLA_net"], label="ILLA", color="C0")
    axes[1].plot(seed201["OLLA_net"], label="OLLA", color="C1")
    axes[1].set_ylabel("delivered − drops\n(bits/RE)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    for name, color in (("ILLA", "C0"), ("OLLA", "C1")):
        slots, q = zip(*queues[name])
        axes[2].plot(slots, q, label=name, color=color, lw=0.8)
    axes[2].axhline(env.queue_capacity, color="0.5", ls=":", lw=0.8)
    axes[2].set_xlabel("slot")
    axes[2].set_ylabel("queue (bits)")
    axes[2].set_ylim(0, env.queue_capacity * 1.05)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "illa_olla_arr300_500_seed201.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
