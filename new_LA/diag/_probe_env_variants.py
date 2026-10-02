# Where does OLLA leave room? For each candidate environment, compare OLLA with a
# bias oracle (knows the current CQI bias) and the true-SINR genie.
# Bias variants are injected into env._sinr_fb_db after reset; la_env is unchanged.

import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sionna.phy.utils import db_to_lin
from sionna.sys import PHYAbstraction

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from cqi import mcs_from_sinr_db, tb_layout_from_mcs, tbler_from_phy
from la_env import DownlinkLAEnv, seed_phy
from policies import make_baseline_policy

SEEDS = (201, 202, 203, 204, 205)


def bias_fixed(seed, t):
    return np.full(t.shape, 3.0)


def bias_per_episode(seed, t):
    return np.full(t.shape, np.random.default_rng(seed + 7).uniform(0.0, 6.0))


def bias_drift(seed, t):
    phase = np.random.default_rng(seed + 7).uniform(0, 2 * np.pi)
    return 3.0 + 3.0 * np.sin(2 * np.pi * t / 400.0 + phase)


VARIANTS = [
    ("V0 bias +3 fixed (current)", {}, bias_fixed),
    ("V1 bias ~U[0,6] per episode", {}, bias_per_episode),
    ("V2 bias 3+3sin(t/400)", {}, bias_drift),
    ("V3 mean SINR jumps", {"sinr_mean_range_db": [0.0, 20.0], "sinr_mean_change_prob": 0.005}, bias_fixed),
    ("V4 CQI delay 1-4 per ep", {"cqi_delay_slots": None}, bias_fixed),
]


def reset(env, seed, bias_fn):
    t = np.arange(env.num_slots)
    bias = bias_fn(seed, t)
    env.cqi_bias_db = 0.0
    env.reset(seed=seed)
    env._sinr_fb_db = np.clip(env._sinr_fb_db + bias, env.sinr_min_db, env.sinr_max_db)
    env._probe_bias = bias
    cqi = env._report_cqi_at(env._t)
    return env._state(cqi), env._info(cqi_index=cqi)


def run(env, pol, seed, bias_fn):
    if hasattr(pol, "reset"):
        pol.reset()
    seed_phy(seed)
    state, info = reset(env, seed, bias_fn)
    ret, done = 0.0, False
    while not done:
        state, r, term, trunc, info = env.step(int(pol(state, info)))
        ret += r
        done = term or trunc
    return ret


def main():
    cfg0 = yaml.safe_load((ROOT / "configs" / "downlink_la.yaml").open(encoding="utf-8"))
    phy = PHYAbstraction()
    bler = float(cfg0["bler_target"])

    for vname, over, bias_fn in VARIANTS:
        cfg = dict(cfg0, **over)
        env = DownlinkLAEnv.from_config(cfg, phy_abs=phy)
        tbs = {m: int(tb_layout_from_mcs(m, env._num_re_t)[0].reshape(-1)[0].item())
               for m in range(env.mcs_min, env.mcs_max + 1)}
        mcs_all = torch.arange(env.mcs_min, env.mcs_max + 1, dtype=torch.int32)
        cal = make_baseline_policy("olla", env, bler_target=bler)._cqi_to_sinr_db
        gap = env.harq_retx_gap_slots + 1

        def shifted(shift_fn, target=0.05):
            def pol(state, info):
                s = cal[int(info["cqi_index"])] - shift_fn(info)
                return mcs_from_sinr_db(phy, s, env._num_re_t, env.mcs_min, env.mcs_max, bler_target=target) - env.mcs_min
            return pol

        def genie(state, info):
            lin = db_to_lin(torch.tensor([info["sinr_true_db"]], dtype=torch.float32))
            p = tbler_from_phy(phy, mcs_all, lin, env._num_re_t)[0].numpy().reshape(-1)
            pay = np.minimum([tbs[int(m)] for m in mcs_all], info["queue"])
            return int(np.argmax(pay * (1 - p) / (1 + gap * p)))

        policies = [
            ("ILLA", make_baseline_policy("illa", env, bler_target=bler)),
            ("OLLA step 0.1", make_baseline_policy("olla", env, bler_target=bler, olla_step_up_db=0.1)),
            ("OLLA step 0.3", make_baseline_policy("olla", env, bler_target=bler, olla_step_up_db=0.3)),
            ("fixed CQI-4dB", shifted(lambda info: 4.0)),
            ("bias oracle", shifted(lambda info: env._probe_bias[min(info["slot"], env.num_slots - 1)] + 1.0)),
            ("genie", genie),
        ]
        print(f"\n== {vname}", flush=True)
        base = None
        for pname, pol in policies:
            r = np.asarray([run(env, pol, s, bias_fn) for s in SEEDS])
            if pname == "OLLA step 0.1":
                base = r.mean()
            rel = "" if base is None else f"  vs OLLA {100 * (r.mean() / base - 1):+5.1f}%"
            print(f"  {pname:15s} {r.mean():7.1f} ±{r.std():5.1f}{rel}", flush=True)


if __name__ == "__main__":
    main()
