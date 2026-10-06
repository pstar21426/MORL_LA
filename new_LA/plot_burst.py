# Plots for the burst setting: training curve, paired eval, slot traces,
# MCS-vs-CQI maps, and burst-on vs burst-off statistics for ILLA / OLLA / DDQN.
# The burst mask is recovered from a second env with burst_atten_db = 0 on the
# same seed (clean SINR minus decoding SINR).

import argparse
import copy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sionna.sys import PHYAbstraction

from ddqn import DDQNAgent
from la_env import DownlinkLAEnv, seed_phy
from policies import make_baseline_policy
from run_la_sim import load_config

HERE = Path(__file__).resolve().parent
COLORS = {"ILLA": "C0", "OLLA": "C1", "DDQN": "C2"}
ACK_COLS = slice(7, 10)


class Greedy:
    def __init__(self, agent):
        self.agent = agent

    def reset(self):
        pass

    def __call__(self, state, info):
        del info
        return self.agent.select_action(state, greedy=True)


def load_agent(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    agent = DDQNAgent(int(ck["state_dim"]), int(ck["n_actions"]), hidden=int(ck.get("hidden", 256)))
    agent.q.load_state_dict(ck["q"])
    agent.q.eval()
    return agent


def rollout(env, policy, seed):
    seed_phy(seed)
    state, info = env.reset(seed=seed)
    policy.reset()
    rows = []
    done = False
    while not done:
        q_before = env._q
        action = policy(state, info)
        cqi = int(info["cqi_index"])
        acks = state[ACK_COLS]
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        out = info["outcome"]
        idle = bool(out.get("idle"))
        rows.append(
            dict(
                slot=min(int(out["decision_slot"]), env.num_slots - 1),
                idle=idle,
                tau=1 if idle else max(int(out["num_slots"]), 1),
                mcs=int(out["mcs_used"]),
                ack=int(out["ack"]),
                reward=float(reward),
                queue=float(q_before),
                cqi=cqi,
                nacks=int((acks == 0).sum()),
            )
        )
        state = next_state
    res = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
    res["sinr_dec"] = np.asarray(env._sinr_true_db, dtype=np.float64)
    return res


def burst_mask(clean_env, env, seed):
    clean_env.reset(seed=seed)
    env.reset(seed=seed)
    return (np.asarray(clean_env._sinr_true_db) - np.asarray(env._sinr_true_db)) > 1e-9, np.asarray(
        clean_env._sinr_true_db, dtype=np.float64
    )


def slot_reward(res, n_slots):
    r = np.zeros(n_slots)
    end = np.minimum(res["slot"] + res["tau"] - 1, n_slots - 1)
    np.add.at(r, end, res["reward"])
    return r


def step_hold(res, key, n_slots):
    out = np.full(n_slots, np.nan)
    for s, tau, idle, v in zip(res["slot"], res["tau"], res["idle"], res[key]):
        if not idle:
            out[s : min(s + tau, n_slots)] = v
    return out


def shade(ax, mask):
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        ax.axvspan(a - 0.5, b - 0.5, color="0.85", lw=0, zorder=0)


def rolling(x, w):
    if len(x) < w:
        return np.asarray(x, dtype=float)
    k = np.ones(w) / w
    return np.convolve(x, k, mode="valid")


def plot_training(log, ref, out):
    ep = log["episode"]
    fig, axs = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    panels = [
        ("return", "episode return"),
        ("first_tx_bler", "first-tx BLER"),
        ("mean_mcs", "mean MCS"),
        ("epsilon", "epsilon"),
    ]
    for ax, (key, label) in zip(axs, panels):
        y = log[key]
        ax.plot(ep, y, color="C2", alpha=0.35, lw=0.8, label="DDQN (per episode)")
        w = 10
        if len(y) >= w:
            ax.plot(ep[w - 1 :], rolling(y, w), color="C2", lw=1.8, label=f"DDQN ({w}-ep mean)")
        for name in ("OLLA", "ILLA"):
            if key in ref.get(name, {}):
                xs, ys = ref[name]["seeds"] + 1, ref[name][key]
                ax.plot(xs, ys, "o", ms=3, color=COLORS[name], alpha=0.6)
                ax.axhline(np.mean(ys), color=COLORS[name], ls="--", lw=1.2, label=f"{name} (same train seeds)")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    if "rho" in log and np.isfinite(log["rho"]).any():
        axs[0].plot(ep, 1000.0 * log["rho"], color="k", lw=1.2, label="rho x 1000")
        title = "DDQN training, learner reward r - rho*tau, burst on"
    else:
        title = "DDQN training, learner reward r, burst on"
    axs[0].legend(fontsize=8, loc="lower right")
    axs[-1].set_xlabel("training episode (env seed = episode - 1)")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_paired(seeds, ret, out):
    names = ["ILLA", "OLLA", "DDQN"]
    x = np.arange(len(seeds))
    w = 0.27
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw=dict(width_ratios=[2, 1]))
    for i, n in enumerate(names):
        axs[0].bar(x + (i - 1) * w, ret[n], w, color=COLORS[n], label=n)
    axs[0].set_xticks(x, [str(s) for s in seeds])
    axs[0].set_xlabel("eval seed")
    axs[0].set_ylabel("episode return")
    axs[0].set_ylim(min(min(v) for v in ret.values()) * 0.85, None)
    axs[0].legend(fontsize=8)
    axs[0].grid(True, axis="y", alpha=0.3)

    d = np.asarray(ret["DDQN"]) - np.asarray(ret["OLLA"])
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
    axs[1].bar(x, d, color=np.where(d >= 0, "C2", "C3"))
    axs[1].axhline(0, color="k", lw=0.8)
    axs[1].axhline(d.mean(), color="k", ls="--", lw=1.0, label=f"mean {d.mean():+.1f} (SE {se:.1f})")
    axs[1].set_xticks(x, [str(s) for s in seeds])
    axs[1].set_xlabel("eval seed")
    axs[1].set_ylabel("DDQN - OLLA")
    axs[1].legend(fontsize=8)
    axs[1].grid(True, axis="y", alpha=0.3)
    fig.suptitle("Paired eval (same seeds for all policies)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def pick_window(mask, length, n_slots):
    on = np.flatnonzero(mask)
    start = int(on[0]) - 30 if on.size else 0
    for s in on:
        seg = mask[max(s - 30, 0) : max(s - 30, 0) + length]
        if seg.sum() >= 0.2 * length and np.diff(seg.astype(int)).clip(0).sum() >= 2:
            start = int(s) - 30
            break
    start = int(np.clip(start, 0, n_slots - length))
    return start, start + length


def plot_trace(seed, mask, clean, runs, n_slots, out, window=None):
    t = np.arange(n_slots)
    lo, hi = window if window else (0, n_slots)
    fig, axs = plt.subplots(4, 1, figsize=(11, 10), sharex=True)

    axs[0].plot(t, clean, color="0.4", lw=0.9, ls="--", label="SINR seen by CQI (no burst)")
    axs[0].plot(t, runs["OLLA"]["sinr_dec"], color="k", lw=1.0, label="decoding SINR")
    axs[0].set_ylabel("SINR [dB]")
    axs[0].legend(fontsize=8, loc="lower right")

    for n in ("OLLA", "DDQN"):
        axs[1].step(t, step_hold(runs[n], "mcs", n_slots), where="post", color=COLORS[n], lw=1.1, label=n)
        r = runs[n]
        nack = (~r["idle"]) & (r["ack"] == 0)
        sel = nack & (r["slot"] >= lo) & (r["slot"] < hi)
        axs[1].plot(r["slot"][sel], r["mcs"][sel], "x", color=COLORS[n], ms=5)
    axs[1].set_ylabel("MCS (x = first-tx NACK)")
    axs[1].legend(fontsize=8, loc="upper right")

    for n in ("OLLA", "DDQN"):
        q = np.full(n_slots, np.nan)
        q[runs[n]["slot"]] = runs[n]["queue"]
        idx = ~np.isnan(q)
        axs[2].plot(t[idx], q[idx] / 1000.0, color=COLORS[n], lw=1.0, label=n)
    axs[2].set_ylabel("queue [kbit]")
    axs[2].legend(fontsize=8, loc="upper right")

    for n in ("ILLA", "OLLA", "DDQN"):
        cum = np.cumsum(slot_reward(runs[n], n_slots))
        if n == "OLLA":
            base = cum
        axs[3].plot(t, cum, color=COLORS[n], lw=1.2, label=n)
    if window:
        axs[3].set_ylim(np.nanmin([np.cumsum(slot_reward(runs[n], n_slots))[lo] for n in runs]) - 20,
                        np.nanmax([np.cumsum(slot_reward(runs[n], n_slots))[hi - 1] for n in runs]) + 20)
    axs[3].set_ylabel("cumulative return")
    axs[3].set_xlabel("slot")
    axs[3].legend(fontsize=8, loc="upper left")
    ax2 = axs[3].twinx()
    ax2.plot(t, np.cumsum(slot_reward(runs["DDQN"], n_slots)) - base, color="C3", lw=0.9, alpha=0.8)
    ax2.axhline(0, color="C3", lw=0.5, ls=":")
    ax2.set_ylabel("DDQN - OLLA", color="C3")

    for ax in axs:
        shade(ax, mask)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(lo, hi - 1)
    tag = "" if window is None else f", slots {lo}-{hi - 1}"
    fig.suptitle(f"seed {seed}{tag} (grey = burst on, -8 dB)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_policy_map(pooled, out):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    for n in ("ILLA", "OLLA", "DDQN"):
        r = pooled[n]
        busy = ~r["idle"]
        cq, m = r["cqi"][busy], r["mcs"][busy]
        xs = np.unique(cq)
        axs[0].plot(xs, [m[cq == c].mean() for c in xs], "o-", color=COLORS[n], ms=4, label=n)
    axs[0].set_title("mean MCS vs reported CQI")
    axs[0].set_xlabel("CQI")
    axs[0].set_ylabel("MCS")
    axs[0].legend(fontsize=8)

    for ax, n in zip(axs[1:], ("OLLA", "DDQN")):
        r = pooled[n]
        busy = ~r["idle"]
        for k, ls in zip((0, 1, 2), ("-", "--", ":")):
            sel = busy & (np.minimum(r["nacks"], 2) == k)
            cq, m = r["cqi"][sel], r["mcs"][sel]
            xs = [c for c in np.unique(cq) if (cq == c).sum() >= 15]
            if xs:
                lab = f"{k}{'+' if k == 2 else ''} NACK in last 3 (n={sel.sum()})"
                ax.plot(xs, [m[cq == c].mean() for c in xs], "o", ls=ls, color=COLORS[n], ms=4, label=lab)
        ax.set_title(f"{n}: MCS vs CQI by recent first-tx NACKs")
        ax.set_xlabel("CQI")
        ax.legend(fontsize=8)
    for ax in axs:
        ax.grid(True, alpha=0.3)
    fig.suptitle("Policy maps, pooled over eval seeds (cells with >= 15 decisions)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_burst_split(pooled, masks, out):
    names = ["ILLA", "OLLA", "DDQN"]
    stats = {}
    for n in names:
        r = pooled[n]
        on = masks[n][r["slot"]]
        busy = ~r["idle"]
        s = {}
        for key, sel in (("off", busy & ~on), ("on", busy & on)):
            s[key] = dict(
                bler=1.0 - r["ack"][sel].mean(),
                mcs=r["mcs"][sel].mean(),
                rate=r["reward"][sel].sum() / max(r["tau"][sel].sum(), 1),
                tau=r["tau"][sel].mean(),
            )
        stats[n] = s
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    panels = [
        ("rate", "return per occupied slot"),
        ("bler", "first-tx BLER"),
        ("mcs", "mean MCS"),
        ("tau", "slots per TB"),
    ]
    x = np.arange(2)
    w = 0.27
    for ax, (key, label) in zip(axs, panels):
        for i, n in enumerate(names):
            v = [stats[n]["off"][key], stats[n]["on"][key]]
            ax.bar(x + (i - 1) * w, v, w, color=COLORS[n], label=n)
        ax.set_xticks(x, ["burst off", "burst on"])
        ax.set_title(label)
        ax.grid(True, axis="y", alpha=0.3)
    axs[0].legend(fontsize=8)
    fig.suptitle("TBs grouped by burst state at the decision slot (pooled eval seeds)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=HERE / "configs" / "downlink_la.yaml")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--train-ref-every", type=int, default=10)
    args = p.parse_args()

    cfg = load_config(args.config)
    out_root = args.out_dir or HERE / cfg.get("out_dir", "outputs")
    seed = int(cfg.get("seed", 0))
    plot_dir = out_root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    phy = PHYAbstraction()
    env = DownlinkLAEnv.from_config(cfg, phy_abs=phy)
    clean_cfg = copy.deepcopy(cfg)
    clean_cfg["burst_atten_db"] = 0.0
    clean_env = DownlinkLAEnv.from_config(clean_cfg, phy_abs=phy)
    n_slots = env.num_slots
    bler = float(cfg["bler_target"])

    policies = {
        "ILLA": make_baseline_policy("illa", env, bler_target=bler),
        "OLLA": make_baseline_policy("olla", env, bler_target=bler, olla_step_up_db=cfg.get("olla_step_up_db")),
        "DDQN": Greedy(load_agent(out_root / f"ddqn_seed{seed}.pt")),
    }

    log = dict(np.load(out_root / f"ddqn_train_log_seed{seed}.npz"))
    ref = {}
    ref_seeds = np.arange(seed, seed + len(log["episode"]), args.train_ref_every)
    for n in ("OLLA", "ILLA"):
        rows = [rollout(env, policies[n], int(s)) for s in ref_seeds]
        ref[n] = dict(
            seeds=ref_seeds - seed,
            **{"return": np.array([r["reward"].sum() for r in rows])},
            first_tx_bler=np.array([1.0 - r["ack"][~r["idle"]].mean() for r in rows]),
            mean_mcs=np.array([r["mcs"][~r["idle"]].mean() for r in rows]),
        )
        print(f"train-seed ref {n}: mean return {ref[n]['return'].mean():.1f}")
    plot_training(log, ref, plot_dir / "train_curve.png")

    ev = np.load(out_root / f"ddqn_eval_seeds{seed}.npz")
    eval_seeds = [int(s) for s in ev["eval_seeds"]]
    runs = {s: {n: rollout(env, pol, s) for n, pol in policies.items()} for s in eval_seeds}
    ret = {n: [runs[s][n]["reward"].sum() for s in eval_seeds] for n in policies}
    for n in policies:
        print(f"eval {n}: {np.mean(ret[n]):.1f}  (npz {float(np.mean(ev[n.lower() + '_return'])):.1f})")
    plot_paired(eval_seeds, ret, plot_dir / "eval_paired.png")

    masks = {}
    for s in eval_seeds:
        masks[s], clean = burst_mask(clean_env, env, s)
        if s == eval_seeds[0]:
            plot_trace(s, masks[s], clean, runs[s], n_slots, plot_dir / f"trace_seed{s}.png")
            win = pick_window(masks[s], 200, n_slots)
            plot_trace(s, masks[s], clean, runs[s], n_slots, plot_dir / f"trace_seed{s}_zoom.png", window=win)

    pooled, pooled_mask = {}, {}
    for n in policies:
        parts = [runs[s][n] for s in eval_seeds]
        offs = np.cumsum([0] + [n_slots] * (len(eval_seeds) - 1))
        pooled[n] = {k: np.concatenate([p[k] + o if k == "slot" else p[k] for p, o in zip(parts, offs)])
                     for k in parts[0] if k != "sinr_dec"}
        pooled_mask[n] = np.concatenate([masks[s] for s in eval_seeds])
    plot_policy_map(pooled, plot_dir / "policy_map.png")
    stats = plot_burst_split(pooled, pooled_mask, plot_dir / "burst_split.png")
    for n, s in stats.items():
        print(
            f"{n}: off rate {s['off']['rate']:.3f} bler {s['off']['bler']:.3f} mcs {s['off']['mcs']:.1f} | "
            f"on rate {s['on']['rate']:.3f} bler {s['on']['bler']:.3f} mcs {s['on']['mcs']:.1f}"
        )
    print(f"Saved plots -> {plot_dir}")


if __name__ == "__main__":
    main()
