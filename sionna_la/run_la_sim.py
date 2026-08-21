from __future__ import annotations
import argparse
import yaml
from typing import Any




from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from sionna.phy import config as sionna_config
from sionna.phy.nr.utils import decode_mcs_index
from sionna.phy.utils import db_to_lin, lin_to_db
from sionna.sys import (
    InnerLoopLinkAdaptation,
    OuterLoopLinkAdaptation,
    PHYAbstraction,
)

from channel import add_cqi_noise, generate_sinr_db_trace


def load_config(path: Path):
    # dict[str, Any], from downlink_la.yaml(parameters)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
"""        
    f = path.open("r", encoding="utf-8")
    data = yaml.safe_load(f)
    f.close()
    return data
"""


def spectral_efficiency(mcs_index: torch.Tensor, mcs_table_index: int):
    # float, SE [bps/Hz] = modulation_order * coderate for PDSCH MCS.
    # mcs_index: int, MCS index chosen by ILLA or OLLA
    # mcs_table_index: int, MCS table index
    mod_order, coderate = decode_mcs_index(
        mcs_index,
        table_index=mcs_table_index,
        is_pusch=False, # downlink
    )
    return float((mod_order * coderate).item())
    # (bits / symbol) * coderate

"""
One slot loop:
    observe γ̂ → choose MCS → PHYAbs(true γ, MCS) → ACK → update SE/BLER logs
"""
def run_controller(
    name: str, # "illa" or "olla"
    controller: Any, # InnerLoopLinkAdaptation or OuterLoopLinkAdaptation
    sinr_true_db: np.ndarray,
    sinr_fb_db: np.ndarray, # noisy CQI
    num_allocated_re: int, # number of allocated REs
    mcs_table_index: int, # MCS table index
    mcs_category: int, # MCS category
    phy_abs: PHYAbstraction, # PHYAbstraction
): # dict[str, np.ndarray]
    
    t_slots = len(sinr_true_db) # number of slots
    mcs_hist = np.zeros(t_slots, dtype=np.int32) # MCS index history
    harq_hist = np.zeros(t_slots, dtype=np.int32) # HARQ history
    se_hist = np.zeros(t_slots, dtype=np.float64) # SE history
    tbler_hist = np.zeros(t_slots, dtype=np.float64) # BLER history
    bits_hist = np.zeros(t_slots, dtype=np.int32) # decoded bits history

    num_re = torch.tensor([num_allocated_re], dtype=torch.int32) # number of allocated REs
    harq = torch.tensor([-1], dtype=torch.int32)  # missing at t=0

    for t in range(t_slots):
        sinr_fb = db_to_lin(torch.tensor([float(sinr_fb_db[t])], dtype=torch.float32))
        sinr_true = db_to_lin(torch.tensor([float(sinr_true_db[t])], dtype=torch.float32))

        # agent chooses MCS from feedback only
        # agent observes sinr_eff, not sinr_true
        if name == "illa":
            mcs = controller(
                num_allocated_re=num_re,
                sinr_eff=sinr_fb,
                mcs_table_index=mcs_table_index,
                mcs_category=mcs_category,
            )
        else:  # olla
            mcs = controller(
                num_allocated_re=num_re,
                sinr_eff=sinr_fb,
                mcs_table_index=mcs_table_index,
                mcs_category=mcs_category,
                harq_feedback=harq,
            )

        # --- nature / PHY uses TRUE SINR ---
        num_decoded_bits, harq, _sinr_eff, tbler, _bler = phy_abs(
            mcs,
            sinr_eff=sinr_true,
            num_allocated_re=num_re,
            mcs_table_index=mcs_table_index,
            mcs_category=mcs_category,
        )

        ack = int(harq)  # 1 ACK, 0 NACK
        se = spectral_efficiency(mcs, mcs_table_index) if ack == 1 else 0.0

        mcs_hist[t] = int(mcs.item())
        harq_hist[t] = ack
        se_hist[t] = se
        tbler_hist[t] = float(tbler.item())
        bits_hist[t] = int(num_decoded_bits.item())

    return {
        "mcs": mcs_hist,
        "harq": harq_hist,
        "se": se_hist,
        "tbler": tbler_hist,
        "decoded_bits": bits_hist,
    }


def plot_results(
    sinr_true_db: np.ndarray,
    sinr_fb_db: np.ndarray,
    results: dict[str, dict[str, np.ndarray]],
    bler_target: float,
    out_path: Path,
) -> None:
    slots = np.arange(len(sinr_true_db))
    fig, axs = plt.subplots(4, 1, figsize=(9, 10), sharex=True)

    axs[0].plot(slots, sinr_true_db, label="true SINR", color="C0")
    axs[0].plot(slots, sinr_fb_db, ":", label="noisy CQI", color="C1", alpha=0.85)
    axs[0].set_ylabel("SINR [dB]")
    axs[0].legend(loc="best")
    axs[0].grid(True, alpha=0.3)
    axs[0].set_title("5G DL LA sandbox (Sionna PHYAbstraction + ILLA/OLLA)")

    for name, style in (("illa", "-"), ("olla", "-.")):
        axs[1].plot(slots, results[name]["mcs"], style, label=name.upper())
    axs[1].set_ylabel("MCS index")
    axs[1].legend(loc="best")
    axs[1].grid(True, alpha=0.3)

    for name, style in (("illa", "-"), ("olla", "-.")):
        axs[2].plot(slots, results[name]["se"], style, label=name.upper())
    axs[2].set_ylabel("SE [bps/Hz]")
    axs[2].legend(loc="best")
    axs[2].grid(True, alpha=0.3)

    for name, style in (("illa", "-"), ("olla", "-.")):
        harq = results[name]["harq"].astype(np.float64)
        emp_bler = 1.0 - np.cumsum(harq) / np.arange(1, len(harq) + 1)
        axs[3].plot(slots, emp_bler, style, label=f"{name.upper()} empirical BLER")
    axs[3].axhline(bler_target, color="k", ls="--", label="BLER target")
    axs[3].set_ylabel("Empirical BLER")
    axs[3].set_xlabel("Slot")
    axs[3].legend(loc="best")
    axs[3].grid(True, alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def summarize(name: str, res: dict[str, np.ndarray], bler_target: float) -> None:
    emp_bler = 1.0 - float(np.mean(res["harq"]))
    mean_se = float(np.mean(res["se"]))
    mean_mcs = float(np.mean(res["mcs"]))
    print(
        f"[{name.upper()}] mean SE={mean_se:.3f} bps/Hz | "
        f"emp BLER={emp_bler:.3f} (target {bler_target}) | "
        f"mean MCS={mean_mcs:.1f} | "
        f"mean decoded bits/slot={float(np.mean(res['decoded_bits'])):.1f}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sionna 5G DL LA (ILLA vs OLLA)")
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "downlink_la.yaml",
    )
    p.add_argument("--num-slots", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.num_slots is not None:
        cfg["num_slots"] = args.num_slots
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.out_dir is not None:
        cfg["out_dir"] = str(args.out_dir)

    seed = int(cfg["seed"])
    sionna_config.seed = seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_slots = int(cfg["num_slots"])
    bler_target = float(cfg["bler_target"])
    mcs_table_index = int(cfg["mcs_table_index"])
    mcs_category = int(cfg["mcs_category"])
    num_re = int(cfg["num_allocated_re"])

    print("=== Generating correlated effective SINR ===")
    sinr_true_db = generate_sinr_db_trace(
        num_slots=num_slots,
        mean_db=float(cfg["sinr_mean_db"]),
        rho=float(cfg["sinr_ar_rho"]),
        innov_std_db=float(cfg["sinr_innov_std_db"]),
        sinr_min_db=float(cfg["sinr_min_db"]),
        sinr_max_db=float(cfg["sinr_max_db"]),
        seed=seed,
    )
    sinr_fb_db = add_cqi_noise(
        sinr_true_db,
        noise_std_db=float(cfg["cqi_noise_std_db"]),
        delay_slots=int(cfg["cqi_delay_slots"]),
        seed=seed + 1,
    )

    print("=== Building Sionna PHYAbstraction / ILLA / OLLA ===")
    phy_abs = PHYAbstraction()
    illa = InnerLoopLinkAdaptation(phy_abs, bler_target=bler_target)
    olla = OuterLoopLinkAdaptation(phy_abs, num_ut=1, bler_target=bler_target)

    print("=== Running ILLA ===")
    res_illa = run_controller(
        "illa", illa, sinr_true_db, sinr_fb_db, num_re, mcs_table_index, mcs_category, phy_abs
    )
    summarize("illa", res_illa, bler_target)

    print("=== Running OLLA ===")
    # fresh OLLA state
    olla = OuterLoopLinkAdaptation(phy_abs, num_ut=1, bler_target=bler_target)
    res_olla = run_controller(
        "olla", olla, sinr_true_db, sinr_fb_db, num_re, mcs_table_index, mcs_category, phy_abs
    )
    summarize("olla", res_olla, bler_target)

    out_dir = Path(cfg["out_dir"])
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {"illa": res_illa, "olla": res_olla}

    if cfg.get("save_plot", True):
        plot_path = out_dir / f"la_illa_vs_olla_seed{seed}.png"
        plot_results(sinr_true_db, sinr_fb_db, results, bler_target, plot_path)
        print(f"Saved plot → {plot_path}")

    if cfg.get("save_npz", True):
        npz_path = out_dir / f"la_illa_vs_olla_seed{seed}.npz"
        np.savez_compressed(
            npz_path,
            sinr_true_db=sinr_true_db,
            sinr_fb_db=sinr_fb_db,
            illa_mcs=res_illa["mcs"],
            illa_harq=res_illa["harq"],
            illa_se=res_illa["se"],
            illa_tbler=res_illa["tbler"],
            olla_mcs=res_olla["mcs"],
            olla_harq=res_olla["harq"],
            olla_se=res_olla["se"],
            olla_tbler=res_olla["tbler"],
            bler_target=bler_target,
            seed=seed,
        )
        print(f"Saved npz  → {npz_path}")

    print("Done. Next: read the SINR / MCS / BLER plots, then formalize as MDP.")


if __name__ == "__main__":
    main()
