# Recollect the offline set with terminated / truncated split and check that
# only the done flags changed. Also checks ObsProjector is identity on real states.

import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sionna.sys import PHYAbstraction

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from cqi import CQI_MAX
from la_env import DownlinkLAEnv
from train_mopo import (
    DEVICE, ObsProjector, apply_norm, collect_dataset, load_dataset, normalize_fit, save_dataset,
)


def main():
    path = ROOT / "datasets" / "offline_illa_olla.npz"
    old, meta = load_dataset(path)
    cfg = yaml.safe_load((ROOT / "configs" / "downlink_la.yaml").open(encoding="utf-8"))
    env = DownlinkLAEnv.from_config(cfg, phy_abs=PHYAbstraction())

    mean, std = normalize_fit(old["observations"])
    proj = ObsProjector(mean, std, env.hist.num_lags, CQI_MAX, env._mcs_span)
    for key in ("observations", "next_observations"):
        x = torch.as_tensor(apply_norm(old[key], mean, std), device=DEVICE)
        err = (proj(x) - x).abs().max().item()
        print(f"projector max |change| on real {key}: {err:.2e}")

    new = collect_dataset(
        env, int(meta["episodes_per_policy"]), float(meta["epsilon"]), int(meta["seed0"]),
        float(cfg["bler_target"]), cfg.get("olla_step_up_db"),
    )
    for key in ("observations", "actions", "rewards", "next_observations", "slots"):
        print(f"{key:18s} identical: {np.array_equal(old[key], new[key])}")
    print(f"old done=1: {int(old['terminations'].sum())}  new terminations: "
          f"{int(new['terminations'].sum())}  new truncations: {int(new['truncations'].sum())}")
    print(f"old done == new truncations: {np.array_equal(old['terminations'], new['truncations'])}")

    shutil.copy2(path, path.with_suffix(".npz.bak"))
    meta["done_note"] = "terminations = env terminated only; time-limit ends are in truncations"
    save_dataset(path, new, meta)


if __name__ == "__main__":
    main()
