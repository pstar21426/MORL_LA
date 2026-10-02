# Offline dataset: coverage per CQI, and per-decision vs per-slot reward by MCS.

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from train_mopo import load_dataset


def main():
    data, meta = load_dataset(ROOT / "datasets" / "offline_illa_olla.npz")
    obs, act, rew, slots, done = (
        data["observations"], data["actions"], data["rewards"], data["slots"], data["terminations"]
    )
    n = len(rew)
    print("meta", meta)
    print(f"N={n}  done=1 count={int(done.sum())}  reward mean={rew.mean():.3f}  "
          f"P(r=0)={np.mean(rew == 0):.3f}  slots mean={slots.mean():.2f}")
    qfrac = obs[:, -1]
    print(f"queue frac p10/50/90 = {np.percentile(qfrac, [10, 50, 90]).round(3)}  "
          f"frac q>=0.95: {np.mean(qfrac >= 0.95):.3f}")

    cqi = np.rint(obs[:, 0] * 15).astype(int)
    mcs = act + 3
    print("\n# coverage: per CQI, #transitions, #distinct MCS, top-2 MCS share, MCS with >=20 samples")
    for q in range(16):
        m = cqi == q
        if m.sum() == 0:
            continue
        vals, cnt = np.unique(mcs[m], return_counts=True)
        order = np.argsort(-cnt)
        top2 = cnt[order[:2]].sum() / m.sum()
        enough = vals[cnt >= 20].tolist()
        print(f"  CQI {q:2d}: n={m.sum():5d} distinct={len(vals):2d} top2={top2:.2f} "
              f"top={[int(v) for v in vals[order[:3]]]} >=20: {enough}")

    print("\n# full-ish queue (q>=0.9): per CQI, MCS -> (n, reward/decision, reward/slot, mean slots)")
    full = qfrac >= 0.9
    for q in (6, 9, 12, 15):
        m = full & (cqi == q)
        if m.sum() == 0:
            continue
        print(f"  CQI {q}:")
        best_dec, best_slot = None, None
        for v in np.unique(mcs[m]):
            k = m & (mcs == v)
            if k.sum() < 15:
                continue
            rd = rew[k].mean()
            rs = rew[k].sum() / slots[k].sum()
            print(f"    MCS {v:2d}: n={k.sum():4d} r/dec={rd:5.2f} r/slot={rs:5.2f} slots={slots[k].mean():.2f}")
            if best_dec is None or rd > best_dec[1]:
                best_dec = (v, rd)
            if best_slot is None or rs > best_slot[1]:
                best_slot = (v, rs)
        print(f"    argmax r/decision -> MCS {best_dec[0]}, argmax r/slot -> MCS {best_slot[0]}")


if __name__ == "__main__":
    main()
