# Buffer, MCS, and state diagnostics. Does not train.
# Replay contents were not saved; episode log + a uniform-action probe stand in.

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sionna.sys import PHYAbstraction

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cqi import tb_layout_from_mcs
from ddqn import DDQNAgent
from la_env import DownlinkLAEnv, seed_phy
from policies import make_baseline_policy

OUT = Path(__file__).resolve().parent / "outputs"
SEEDS = (201, 202, 203)
PROBE_SEEDS = (0, 1)
QUEUE_BINS = (0, 3000, 6000, 9000, 12000, 15000)


def load_config(path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def tbs_table(env):
    table = {}
    for mcs in range(env.mcs_min, env.mcs_max + 1):
        tbs, _, _ = tb_layout_from_mcs(
            mcs,
            env.num_allocated_re,
            mcs_table_index=env.mcs_table_index,
            mcs_category=env.mcs_category,
        )
        table[mcs] = int(tbs.reshape(-1)[0].item())
    return table


def attach_arrival_log(env):
    if not hasattr(env, "_arrive_orig"):
        env._arrive_orig = env._arrive
    log = []
    orig = env._arrive_orig

    def wrapped():
        q_before = int(env._q)
        overflow = int(orig())
        accepted = int(env._q) - q_before
        log.append(
            {
                "slot": int(env._t),
                "q": int(env._q),
                "offered": accepted + overflow,
                "overflow": overflow,
            }
        )
        return overflow

    env._arrive = wrapped
    return log


def rollout(env, choose, seed, name, tbs_of):
    arrivals = attach_arrival_log(env)
    seed_phy(seed)
    state, info = env.reset(seed=seed)
    rows = []
    done = False
    while not done:
        q = int(info["queue"])
        action = int(choose(state, info))
        mcs = int(env.mcs_from_action(action))
        tbs = tbs_of[mcs]
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        out = info["outcome"]
        if not out.get("idle"):
            rows.append(
                {
                    "seed": seed,
                    "policy": name,
                    "slot": int(out["decision_slot"]),
                    "q": q,
                    "q_frac_state": float(state[-1]),
                    "cqi": int(out["cqi_index"]),
                    "cqi_state": float(state[0]),
                    "sinr_true": float(out["sinr_true_db"]),
                    "sinr_hat": float(out["sinr_hat_db"]),
                    "action": action,
                    "mcs": int(out["mcs_used"]),
                    "tbs": tbs,
                    "payload": min(q, tbs),
                    "ack": int(out["ack"]),
                    "success": int(out["tb_success"]),
                    "dropped": int(bool(out["dropped"])),
                    "trunc": int(bool(out["truncated_mid_tb"])),
                    "num_slots": int(out["num_slots"]),
                    "num_retx": int(out["num_retx"]),
                    "overflow": int(out["n_overflow"]),
                    "discard": int(out["discard_bits"]),
                    "delivered_se": float(out["delivered_se"]),
                    "reward": float(reward),
                }
            )
        state = next_state
    return rows, arrivals, int(env._q)


def pack(rows):
    if not rows:
        return {}
    keys = [k for k in rows[0] if k != "policy"]
    out = {k: np.asarray([r[k] for r in rows]) for k in keys}
    out["policy"] = rows[0]["policy"]
    return out


def pct(x, p):
    return float(np.percentile(x, p))


def ledger_ok(rows, arrivals, final_q):
    offered = int(sum(a["offered"] for a in arrivals))
    delivered = int(sum(r["payload"] for r in rows if r["success"]))
    discard = int(sum(r["discard"] for r in rows))
    overflow = int(sum(a["overflow"] for a in arrivals))
    balance = offered - delivered - discard - overflow - final_q
    slots = sorted(a["slot"] for a in arrivals)
    slot_sum = int(sum(r["num_slots"] for r in rows))
    q_ok = all(0 <= a["q"] <= 15000 for a in arrivals)
    unique = len(slots) == len(set(slots)) == 1000 and slots[0] == 0 and slots[-1] == 999
    return {
        "offered": offered,
        "delivered": delivered,
        "discard": discard,
        "overflow": overflow,
        "final_q": final_q,
        "balance": balance,
        "n_arrive": len(arrivals),
        "slot_sum": slot_sum,
        "q_in_range": q_ok,
        "arrive_slots_ok": unique,
    }


def summarize_policy(name, rows):
    q = np.asarray([r["q"] for r in rows], dtype=np.float64)
    mcs = np.asarray([r["mcs"] for r in rows], dtype=np.float64)
    tbs = np.asarray([r["tbs"] for r in rows], dtype=np.float64)
    cqi = np.asarray([r["cqi"] for r in rows], dtype=np.float64)
    ack = np.asarray([r["ack"] for r in rows], dtype=np.float64)
    short = np.asarray([r["q"] < r["tbs"] for r in rows], dtype=np.float64)
    full = np.mean(q >= 15000)
    state_mismatch = sum(
        abs(r["q_frac_state"] * 15000 - r["q"]) > 1.5 for r in rows
    )
    return {
        "n": len(rows),
        "q_mean": float(q.mean()),
        "q_p10": pct(q, 10),
        "q_p50": pct(q, 50),
        "q_p90": pct(q, 90),
        "frac_full": float(full),
        "frac_short": float(short.mean()),
        "mcs_mean": float(mcs.mean()),
        "mcs_p10": pct(mcs, 10),
        "mcs_p50": pct(mcs, 50),
        "mcs_p90": pct(mcs, 90),
        "bler": float(1.0 - ack.mean()),
        "slots": float(np.mean([r["num_slots"] for r in rows])),
        "delivered_se": float(sum(r["delivered_se"] for r in rows)),
        "corr_mcs_q": float(np.corrcoef(mcs, q)[0, 1]),
        "corr_mcs_cqi": float(np.corrcoef(mcs, cqi)[0, 1]),
        "mean_tbs": float(tbs.mean()),
        "state_mismatch": int(state_mismatch),
        "mcs_min": int(mcs.min()),
        "mcs_max": int(mcs.max()),
    }


def mcs_by_queue(rows):
    q = np.asarray([r["q"] for r in rows])
    mcs = np.asarray([r["mcs"] for r in rows], dtype=np.float64)
    ack = np.asarray([r["ack"] for r in rows], dtype=np.float64)
    means, blers, counts = [], [], []
    for lo, hi in zip(QUEUE_BINS[:-1], QUEUE_BINS[1:]):
        if hi == QUEUE_BINS[-1]:
            mask = (q >= lo) & (q <= hi)
        else:
            mask = (q >= lo) & (q < hi)
        counts.append(int(mask.sum()))
        means.append(float(mcs[mask].mean()) if mask.any() else np.nan)
        blers.append(float(1.0 - ack[mask].mean()) if mask.any() else np.nan)
    return means, blers, counts


def save_plots(by_policy, arrivals_201, train):
    colors = {"ILLA": "C0", "OLLA": "C1", "DDQN": "C2", "uniform": "C3"}
    order = ["ILLA", "OLLA", "DDQN"]

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for ax, name in zip(axes, order):
        series = arrivals_201[name]
        ax.plot(series["slot"], series["q"], color=colors[name], lw=0.8)
        ax.axhline(15000, color="0.5", ls=":", lw=0.8)
        ax.set_ylabel("queue (bits)")
        ax.set_title(name)
        ax.set_ylim(0, 16000)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("slot")
    fig.tight_layout()
    fig.savefig(OUT / "queue_seed201.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 15000, 31)
    for name in order:
        q = np.asarray([r["q"] for r in by_policy[name]])
        ax.hist(q, bins=bins, histtype="step", density=True, lw=1.6, label=name, color=colors[name])
    ax.set_xlabel("queue at decision (bits)")
    ax.set_ylabel("density")
    ax.set_title("decision-time buffer, seeds 201-203")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "queue_hist.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    mcs_bins = np.arange(2.5, 29.5, 1.0)
    for name in order:
        mcs = np.asarray([r["mcs"] for r in by_policy[name]])
        axes[0].hist(
            mcs, bins=mcs_bins, histtype="step", density=True, lw=1.6, label=name, color=colors[name]
        )
    axes[0].set_xlabel("MCS")
    axes[0].set_ylabel("density")
    axes[0].set_title("MCS at decision")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    labels = [f"{lo//1000}-{hi//1000}k" for lo, hi in zip(QUEUE_BINS[:-1], QUEUE_BINS[1:])]
    x = np.arange(len(labels))
    width = 0.25
    for i, name in enumerate(order):
        means, _, counts = mcs_by_queue(by_policy[name])
        axes[1].bar(x + (i - 1) * width, means, width, label=name, color=colors[name])
        for j, n in enumerate(counts):
            if n == 0:
                axes[1].text(x[j] + (i - 1) * width, 1, "0", ha="center", fontsize=7, color=colors[name])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_xlabel("queue bin")
    axes[1].set_ylabel("mean MCS")
    axes[1].set_ylim(0, 30)
    axes[1].set_title("MCS vs buffer")
    axes[1].legend()
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "mcs_choice.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for name in order:
        rows = by_policy[name]
        axes[0].hist(
            [r["cqi"] for r in rows],
            bins=np.arange(-0.5, 16.5, 1),
            histtype="step",
            density=True,
            lw=1.4,
            label=name,
            color=colors[name],
        )
        axes[1].hist(
            [r["sinr_true"] for r in rows],
            bins=30,
            histtype="step",
            density=True,
            lw=1.4,
            label=name,
            color=colors[name],
        )
    axes[0].set_title("decision CQI")
    axes[0].set_xlabel("CQI")
    axes[1].set_title("true SINR at decision")
    axes[1].set_xlabel("dB")
    if "uniform" in by_policy:
        uq = [r["q"] / 15000 for r in by_policy["uniform"]]
        dq = [r["q"] / 15000 for r in by_policy["DDQN"]]
        axes[2].hist(uq, bins=20, histtype="step", density=True, lw=1.4, label="uniform", color=colors["uniform"])
        axes[2].hist(dq, bins=20, histtype="step", density=True, lw=1.4, label="DDQN greedy", color=colors["DDQN"])
    axes[2].set_title("queue fraction")
    axes[2].set_xlabel("q / capacity")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "state_dist.png", dpi=120)
    plt.close(fig)

    ep = train["episode"]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    axes[0, 0].plot(ep, train["return"])
    axes[0, 0].set_ylabel("return")
    axes[0, 1].plot(ep, train["first_tx_bler"])
    axes[0, 1].axhline(0.1, color="0.4", ls="--", lw=0.8)
    axes[0, 1].set_ylabel("1st-tx BLER")
    axes[1, 0].plot(ep, train["mean_mcs"])
    axes[1, 0].set_ylabel("mean MCS")
    axes[1, 0].set_xlabel("episode")
    axes[1, 1].plot(ep, train["epsilon"])
    axes[1, 1].set_ylabel("epsilon")
    axes[1, 1].set_xlabel("episode")
    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)
    fig.suptitle("saved training log (episode aggregates)")
    fig.tight_layout()
    fig.savefig(OUT / "train_log.png", dpi=120)
    plt.close(fig)


def fmt_row(name, s):
    return (
        f"{name:8} n={s['n']:4d}  q={s['q_mean']:.0f} "
        f"(p10/50/90 {s['q_p10']:.0f}/{s['q_p50']:.0f}/{s['q_p90']:.0f})  "
        f"full={s['frac_full']:.2f}  q<TBS={s['frac_short']:.2f}  "
        f"MCS={s['mcs_mean']:.1f} ({s['mcs_p10']:.0f}/{s['mcs_p50']:.0f}/{s['mcs_p90']:.0f})  "
        f"BLER={s['bler']:.3f}  slots/TB={s['slots']:.2f}  "
        f"corr(MCS,q)={s['corr_mcs_q']:+.2f}  corr(MCS,CQI)={s['corr_mcs_cqi']:+.2f}"
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "configs" / "downlink_la.yaml")
    phy = PHYAbstraction()
    env = DownlinkLAEnv.from_config(cfg, phy_abs=phy)
    ckpt = torch.load(ROOT / "outputs" / "ddqn_seed0.pt", map_location="cpu", weights_only=False)
    agent = DDQNAgent(
        int(np.prod(env.observation_space.shape)),
        env.action_space.n,
        hidden=int(ckpt["hidden"]),
        epsilon_end=float(ckpt["epsilon_end"]),
    )
    agent.q.load_state_dict(ckpt["q"])
    agent.q.eval()

    illa = make_baseline_policy("illa", env, bler_target=float(cfg["bler_target"]))
    olla = make_baseline_policy(
        "olla",
        env,
        bler_target=float(cfg["bler_target"]),
        olla_step_up_db=cfg.get("olla_step_up_db"),
    )

    def greedy(state, info):
        del info
        return agent.select_action(state, greedy=True)

    probe_rng = np.random.default_rng(0)

    def uniform(state, info):
        del state, info
        return int(probe_rng.integers(0, env.action_space.n))

    tbs_of = tbs_table(env)

    choosers = {"ILLA": illa, "OLLA": olla, "DDQN": greedy}
    by_policy = {k: [] for k in list(choosers) + ["uniform"]}
    ledgers = []
    arrivals_201 = {}

    for name, choose in choosers.items():
        for seed in SEEDS:
            if name == "ILLA":
                illa.reset()
            elif name == "OLLA":
                olla.reset()
            rows, arrivals, final_q = rollout(env, choose, seed, name, tbs_of)
            by_policy[name].extend(rows)
            led = ledger_ok(rows, arrivals, final_q)
            led["policy"] = name
            led["seed"] = seed
            ledgers.append(led)
            if seed == 201:
                arrivals_201[name] = {
                    "slot": np.asarray([a["slot"] for a in arrivals]),
                    "q": np.asarray([a["q"] for a in arrivals]),
                }
            print(
                f"{name} seed {seed}: balance={led['balance']} "
                f"arrive={led['n_arrive']} slots={led['slot_sum']} "
                f"q_ok={led['q_in_range']} slots_ok={led['arrive_slots_ok']}"
            )

    for seed in PROBE_SEEDS:
        rows, arrivals, final_q = rollout(env, uniform, seed, "uniform", tbs_of)
        by_policy["uniform"].extend(rows)
        led = ledger_ok(rows, arrivals, final_q)
        led["policy"] = "uniform"
        led["seed"] = seed
        ledgers.append(led)
        print(
            f"uniform seed {seed}: balance={led['balance']} "
            f"arrive={led['n_arrive']} slots={led['slot_sum']}"
        )

    train = np.load(ROOT / "outputs" / "ddqn_train_log_seed0.npz")
    save_plots(by_policy, arrivals_201, train)

    lines = ["# sim diagnostic", ""]
    lines.append("Seeds 201-203, greedy DDQN from outputs/ddqn_seed0.pt. Replay buffer was not saved.")
    lines.append("")
    lines.append("## ledger")
    bad = [L for L in ledgers if L["balance"] != 0 or not L["q_in_range"] or not L["arrive_slots_ok"] or L["slot_sum"] != 1000]
    lines.append(f"episodes checked: {len(ledgers)}, failures: {len(bad)}")
    for L in ledgers:
        lines.append(
            f"- {L['policy']} seed {L['seed']}: balance {L['balance']}, "
            f"offered {L['offered']}, delivered {L['delivered']}, "
            f"discard {L['discard']}, overflow {L['overflow']}, final_q {L['final_q']}"
        )
    lines.append("")
    lines.append("## decision stats (201-203 pooled)")
    stats = {}
    for name in ("ILLA", "OLLA", "DDQN", "uniform"):
        stats[name] = summarize_policy(name, by_policy[name])
        lines.append(fmt_row(name, stats[name]))
        lines.append(
            f"    state mismatches={stats[name]['state_mismatch']}  "
            f"MCS range {stats[name]['mcs_min']}-{stats[name]['mcs_max']}  "
            f"mean TBS {stats[name]['mean_tbs']:.0f}  "
            f"delivered bits/RE {stats[name]['delivered_se']:.1f}"
        )
    lines.append("")
    lines.append("## MCS by queue bin (mean MCS, 1st-tx BLER, count)")
    for name in ("ILLA", "OLLA", "DDQN"):
        means, blers, counts = mcs_by_queue(by_policy[name])
        bits = []
        for i, (lo, hi) in enumerate(zip(QUEUE_BINS[:-1], QUEUE_BINS[1:])):
            bits.append(f"{lo}-{hi}: MCS {means[i]:.1f} BLER {blers[i]:.2f} n={counts[i]}")
        lines.append(f"- {name}: " + " | ".join(bits))

    ret = train["return"]
    bler = train["first_tx_bler"]
    eps = train["epsilon"]
    lines.append("")
    lines.append("## training log")
    lines.append(
        f"episodes {len(ret)}, return mean {ret.mean():.1f} "
        f"(first 20 {ret[:20].mean():.1f}, last 40 {ret[-40:].mean():.1f}), "
        f"BLER last 40 {bler[-40:].mean():.3f}, "
        f"epsilon end {eps[-1]:.3f}, "
        f"NaN {int(np.isnan(ret).any() or np.isnan(bler).any())}"
    )
    lines.append("")
    lines.append("Figures: queue_seed201.png, queue_hist.png, mcs_choice.png, state_dist.png, train_log.png")

    text = "\n".join(lines) + "\n"
    (OUT / "summary.md").write_text(text, encoding="utf-8")
    print(text)

    np.savez_compressed(
        OUT / "decisions.npz",
        **{
            f"{name}_{k}": np.asarray([r[k] for r in rows])
            for name, rows in by_policy.items()
            for k in ("q", "mcs", "cqi", "sinr_true", "ack", "tbs", "num_slots", "reward")
        },
    )


if __name__ == "__main__":
    main()
