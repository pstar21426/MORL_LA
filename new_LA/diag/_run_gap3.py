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
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    phy = PHYAbstraction()
    env = DownlinkLAEnv.from_config(cfg, phy_abs=phy)
    bler_target = float(cfg["bler_target"])
    n_re = env.num_allocated_re
    print(
        f"gap={env.harq_retx_gap_slots} rho={env.sinr_ar_rho} "
        f"innov={env.sinr_innov_std_db} re={n_re} seeds={SEEDS}"
    )
    illa = make_baseline_policy("illa", env, bler_target=bler_target)
    olla = make_baseline_policy(
        "olla", env, bler_target=bler_target, olla_step_up_db=cfg.get("olla_step_up_db")
    )
    policies = (("ILLA", illa), ("OLLA", olla))
    rows = []
    seed201 = {}
    for seed in SEEDS:
        for name, policy in policies:
            seed_phy(seed)
            res = rollout(env, policy, seed)
            served = res["num_slots"] > 0
            delivered_se = float(res["delivered_se"].sum())
            lost_se = float(res["lost_se"].sum())
            n = int(served.sum())
            row = {
                "seed": seed,
                "name": name,
                "return": float(res["reward"].sum()),
                "delivered_se": delivered_se,
                "bits_per_slot": delivered_se * n_re / env.num_slots,
                "bler": float(1.0 - res["ack"][served].mean()) if n else 0.0,
                "mcs": float(res["mcs_used"][served].mean()) if n else 0.0,
                "slots": float(res["num_slots"][served].mean()) if n else 0.0,
                "tbs": n,
                "lost_se": lost_se,
                "overflow_bits": int(res["n_overflow"].sum()),
            }
            rows.append(row)
            print(
                f"seed {seed} {name}: delivered {delivered_se:.1f} bits/RE "
                f"({row['bits_per_slot']:.1f} bits/slot) return {row['return']:.1f} "
                f"BLER {row['bler']:.3f} MCS {row['mcs']:.1f} "
                f"slots/TB {row['slots']:.2f} TBs {n} "
                f"lost {lost_se:.1f} overflow {row['overflow_bits']}"
            )
            if seed == 201:
                seed201[name] = _slot_cum_return(
                    res["delivered_se"], res["num_slots"], env.num_slots
                )

    print("--- mean ---")
    for name, _ in policies:
        sub = [r for r in rows if r["name"] == name]
        d = np.mean([r["delivered_se"] for r in sub])
        b = np.mean([r["bits_per_slot"] for r in sub])
        ret = np.mean([r["return"] for r in sub])
        bler = np.mean([r["bler"] for r in sub])
        slots = np.mean([r["slots"] for r in sub])
        print(
            f"{name}: delivered {d:.1f} bits/RE ({b:.1f} bits/slot) "
            f"return {ret:.1f} BLER {bler:.3f} slots/TB {slots:.2f}"
        )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(seed201["ILLA"], label="ILLA", color="C0")
    ax.plot(seed201["OLLA"], label="OLLA", color="C1")
    ax.set_xlabel("slot")
    ax.set_ylabel("delivered bits/RE")
    ax.set_title("seed 201, retx gap 3, rho 0.8")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "illa_olla_rho08_seed201.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
