# Mismatch that one scalar OLLA offset cannot represent. Decoding SINR is altered
# after reset; the CQI path (env._sinr_fb_db) still sees the unimpaired SINR + 3 dB.
#   EVM: decoding SINR saturates, 1/g_dec = 1/g + 1/g_evm (g_evm = 20 dB).
#   Bursts: 8 dB interference, Markov on/off (mean on 20 slots, off 80), invisible to CQI.

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
EVM_DB = 20.0
BURST_DB = 8.0


def evm_db(g):
    return -10 * np.log10(10 ** (-g / 10) + 10 ** (-EVM_DB / 10))


def burst_mask(seed, n):
    rng = np.random.default_rng(seed + 11)
    on, out = False, np.zeros(n, dtype=bool)
    for t in range(n):
        on = (rng.random() >= 1 / 20) if on else (rng.random() < 1 / 80)
        out[t] = on
    return out


def make_reset(kind):
    def reset(env, seed):
        env.reset(seed=seed)
        g = env._sinr_true_db
        if kind == "evm":
            env._probe_loss = g - evm_db(g)
            env._sinr_true_db = evm_db(g)
        else:
            m = burst_mask(seed, env.num_slots)
            env._probe_loss = np.where(m, BURST_DB, 0.0)
            env._sinr_true_db = np.clip(g - env._probe_loss, env.sinr_min_db, env.sinr_max_db)
        cqi = env._report_cqi_at(env._t)
        return env._state(cqi), env._info(cqi_index=cqi)
    return reset


def run(env, pol, seed, reset):
    if hasattr(pol, "reset"):
        pol.reset()
    seed_phy(seed)
    state, info = reset(env, seed)
    ret, done = 0.0, False
    while not done:
        state, r, term, trunc, info = env.step(int(pol(state, info)))
        ret += r
        done = term or trunc
    return ret


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "downlink_la.yaml").open(encoding="utf-8"))
    phy = PHYAbstraction()
    env = DownlinkLAEnv.from_config(cfg, phy_abs=phy)
    bler = float(cfg["bler_target"])
    L = env.hist.num_lags
    ack_idx = 1 + 2 * L
    cal = make_baseline_policy("olla", env, bler_target=bler)._cqi_to_sinr_db
    tbs = {m: int(tb_layout_from_mcs(m, env._num_re_t)[0].reshape(-1)[0].item())
           for m in range(env.mcs_min, env.mcs_max + 1)}
    mcs_all = torch.arange(env.mcs_min, env.mcs_max + 1, dtype=torch.int32)
    gap = env.harq_retx_gap_slots + 1

    def from_sinr(s, target=0.05):
        return mcs_from_sinr_db(phy, s, env._num_re_t, env.mcs_min, env.mcs_max, bler_target=target) - env.mcs_min

    def fixed(state, info):
        return from_sinr(cal[int(info["cqi_index"])] - 4.0)

    def evm_map(state, info):
        # CQI-only map that knows the saturation; a learner could find it from (CQI, MCS) data.
        return from_sinr(evm_db(cal[int(info["cqi_index"])] - 3.0) - 1.0)

    def nack_backoff(state, info):
        # Uses only the first-ACK lags already in the state.
        recent_nack = state[ack_idx] == 0 or state[ack_idx + 1] == 0
        return from_sinr(cal[int(info["cqi_index"])] - 4.0 - (BURST_DB if recent_nack else 0.0))

    def loss_oracle(state, info):
        loss = env._probe_loss[min(info["slot"], env.num_slots - 1)]
        return from_sinr(cal[int(info["cqi_index"])] - 4.0 - loss)

    def genie(state, info):
        lin = db_to_lin(torch.tensor([info["sinr_true_db"]], dtype=torch.float32))
        p = tbler_from_phy(phy, mcs_all, lin, env._num_re_t)[0].numpy().reshape(-1)
        pay = np.minimum([tbs[int(m)] for m in mcs_all], info["queue"])
        return int(np.argmax(pay * (1 - p) / (1 + gap * p)))

    def olla(step):
        return make_baseline_policy("olla", env, bler_target=bler, olla_step_up_db=step)

    cases = [
        ("EVM cap 20 dB", "evm", [("OLLA step 0.1", olla(0.1)), ("OLLA step 0.3", olla(0.3)),
                                  ("fixed CQI-4dB", fixed), ("EVM-aware CQI map", evm_map),
                                  ("loss oracle", loss_oracle), ("genie", genie)]),
        ("8 dB bursts", "burst", [("OLLA step 0.1", olla(0.1)), ("OLLA step 0.3", olla(0.3)),
                                  ("fixed CQI-4dB", fixed), ("NACK-lag backoff", nack_backoff),
                                  ("loss oracle", loss_oracle), ("genie", genie)]),
    ]
    for cname, kind, pols in cases:
        reset = make_reset(kind)
        print(f"\n== {cname}", flush=True)
        base = None
        for pname, pol in pols:
            r = np.asarray([run(env, pol, s, reset) for s in SEEDS])
            base = r.mean() if base is None else base
            print(f"  {pname:18s} {r.mean():7.1f} ±{r.std():5.1f}  vs OLLA {100 * (r.mean() / base - 1):+5.1f}%",
                  flush=True)


if __name__ == "__main__":
    main()
