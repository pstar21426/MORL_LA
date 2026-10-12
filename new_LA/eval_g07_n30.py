# Greedy eval of outputs/ddqn_g07_tbs/ddqn_seed1000.pt on seeds 500-529.

import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sionna.sys import PHYAbstraction

from la_env import DownlinkLAEnv
from plot_burst import Greedy, burst_mask, load_agent, rollout
from policies import make_baseline_policy


def _paired_mean_se(diff):
    diff = np.asarray(diff, dtype=np.float64)
    if len(diff) < 2:
        return float(diff.mean()) if len(diff) else float("nan"), float("nan")
    return float(diff.mean()), float(diff.std(ddof=1) / np.sqrt(len(diff)))

HERE = Path(__file__).resolve().parent
CKPT = HERE / "outputs" / "ddqn_g07_tbs" / "ddqn_seed1000.pt"
OUT = CKPT.parent
SEEDS = list(range(500, 530))


def stats(res):
    on = ~res["idle"]
    n = max(int(on.sum()), 1)
    return dict(
        ret=float(res["reward"].sum()),
        bler=float(1.0 - res["ack"][on].mean()) if on.any() else float("nan"),
        mcs=float(res["mcs"][on].mean()) if on.any() else float("nan"),
    )


def burst_stats(res, mask):
    on_tb = ~res["idle"]
    burst = np.asarray([bool(mask[int(s)]) for s in res["slot"]])
    out = {}
    for name, sel in (("off", on_tb & ~burst), ("on", on_tb & burst)):
        if not np.any(sel):
            out[name] = dict(rate=float("nan"), bler=float("nan"), mcs=float("nan"), slots=float("nan"))
            continue
        tau = float(res["tau"][sel].sum())
        out[name] = dict(
            rate=float(res["reward"][sel].sum()) / max(tau, 1.0),
            bler=float(1.0 - res["ack"][sel].mean()),
            mcs=float(res["mcs"][sel].mean()),
            slots=float(res["tau"][sel].mean()),
        )
    return out


def main():
    with (HERE / "configs" / "downlink_la.yaml").open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["queue_feat"] = "tbs"
    phy = PHYAbstraction()
    env = DownlinkLAEnv.from_config(cfg, phy_abs=phy)
    clean_cfg = copy.deepcopy(cfg)
    clean_cfg["burst_atten_db"] = 0.0
    clean = DownlinkLAEnv.from_config(clean_cfg, phy_abs=phy)
    agent, name = load_agent(CKPT, env.num_allocated_re)
    if int(np.prod(env.state_space.shape)) != agent.q.net[0].in_features:
        raise RuntimeError("state dim does not match checkpoint")
    pols = {
        "ILLA": make_baseline_policy("illa", env, bler_target=float(cfg["bler_target"])),
        "OLLA": make_baseline_policy(
            "olla", env, bler_target=float(cfg["bler_target"]), olla_step_up_db=cfg.get("olla_step_up_db")
        ),
        name: Greedy(agent),
    }
    rows = {k: [] for k in pols}
    burst = {k: [] for k in ("OLLA", name)}
    for s in SEEDS:
        mask, _ = burst_mask(clean, env, s)
        for k, pol in pols.items():
            res = rollout(env, pol, s)
            st = stats(res)
            rows[k].append(st)
            if k in burst:
                burst[k].append(burst_stats(res, mask))
        print(
            f"seed {s} {name} {rows[name][-1]['ret']:.1f} OLLA {rows['OLLA'][-1]['ret']:.1f}",
            flush=True,
        )

    d = np.asarray([a["ret"] for a in rows[name]]) - np.asarray([a["ret"] for a in rows["OLLA"]])
    diff, se = _paired_mean_se(d)
    lines = [
        f"# {name} vs ILLA / OLLA, queue_feat=tbs, seeds {SEEDS[0]}-{SEEDS[-1]} (n={len(SEEDS)})",
        "",
        "| | ILLA | OLLA | DDQN | DDQN-OLLA | SE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, label, prec in (
        ("ret", "episode return", 1),
        ("bler", "1st-tx BLER", 3),
        ("mcs", "MCS", 2),
    ):
        a = np.asarray([r[key] for r in rows["ILLA"]])
        b = np.asarray([r[key] for r in rows["OLLA"]])
        c = np.asarray([r[key] for r in rows[name]])
        delta, se_k = _paired_mean_se(c - b)
        lines.append(
            f"| {label} | {a.mean():.{prec}f} | {b.mean():.{prec}f} | {c.mean():.{prec}f} | {delta:+.{prec}f} | {se_k:.{prec}f} |"
        )
    lines += ["", "## Burst split, pooled over seeds", ""]
    lines.append("| | OLLA off | DDQN off | OLLA on | DDQN on |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for key, label, prec in (
        ("rate", "return / slot", 3),
        ("bler", "1st-tx BLER", 3),
        ("mcs", "MCS", 2),
        ("slots", "slots / TB", 2),
    ):
        cells = []
        for who, side in (("OLLA", "off"), (name, "off"), ("OLLA", "on"), (name, "on")):
            cells.append(float(np.nanmean([b[side][key] for b in burst[who]])))
        lines.append(
            f"| {label} | {cells[0]:.{prec}f} | {cells[1]:.{prec}f} | {cells[2]:.{prec}f} | {cells[3]:.{prec}f} |"
        )
    lines += [
        "",
        f"DDQN - OLLA return {diff:+.1f} (SE {se:.1f}), positive seeds {int((d > 0).sum())}/{len(d)}",
        "",
    ]
    text = "\n".join(lines)
    (OUT / "ddqn_eval_table_n30.md").write_text(text, encoding="utf-8")
    print(text, flush=True)

    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    x = np.arange(len(SEEDS))
    ax.bar(x, d, color=np.where(d >= 0, "C2", "C3"))
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(diff, color="k", ls="--", lw=1.0, label=f"mean {diff:+.1f} (SE {se:.1f})")
    ax.set_xticks(x, [str(s) for s in SEEDS], rotation=90, fontsize=7)
    ax.set_ylabel(f"{name} - OLLA")
    ax.set_xlabel("eval seed")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"Paired eval, seeds {SEEDS[0]}-{SEEDS[-1]} (n={len(SEEDS)})", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "plots" / "eval_paired_n30.png", dpi=150)
    plt.close(fig)
    print(f"wrote {OUT / 'ddqn_eval_table_n30.md'}", flush=True)


if __name__ == "__main__":
    main()
