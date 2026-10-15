# Figures and table for the gamma=0.9, 3-train-seed Desideratum 3.2 run.
# Fig 1 uses saved training logs. Fig 2 and the table roll out the checkpoints.

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sionna.sys import PHYAbstraction

from la_env import DownlinkLAEnv, seed_phy
from policies import make_baseline_policy
from uq_ensemble import EnsembleDQNAgent

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "des32_g09_s3_ep120"
TRAIN_SEEDS = [0, 1000, 2000]
EVAL_SEEDS = [205, 206, 207, 208, 209]
MCS_MIN, MCS_MAX = 3, 28
COLORS = {"alea": "C1", "epist": "C0", "OLLA": "C3"}
# Greedy returns already averaged over eval seeds 205-209 in summary.md.
GREEDY = {
    "alea": (1089.5, 1119.5, 1130.6),
    "epist": (1133.1, 1083.1, 1094.3),
}
BINS = (("low 3-11", 3, 11), ("mid 12-20", 12, 20), ("high 21-28", 21, 28))


def load_config():
    with (HERE / "configs" / "downlink_la.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_mode(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    w = ck["nets"][0]["backbone.0.weight"]
    mu_w = ck["nets"][0]["mu.weight"]
    agent = EnsembleDQNAgent(
        int(w.shape[1]),
        int(mu_w.shape[0]),
        n_models=int(ck["n_models"]),
        hidden=int(w.shape[0]),
        gamma=float(ck.get("gamma", 0.7)),
        rho=float(ck["rho"]) if ck.get("rho") is not None else None,
        seed=0,
    )
    for net, state in zip(agent.nets, ck["nets"]):
        net.load_state_dict(state)
        net.eval()
    agent.action_mode = ck.get("action_mode", "mean")
    return agent


def mean_std(vals):
    vals = np.asarray(vals, dtype=np.float64)
    if vals.size == 1:
        return float(vals[0]), float("nan")
    return float(vals.mean()), float(vals.std(ddof=1))


def fmt(mean, std):
    return f"{mean:.3f} ± {std:.3f}"


def fmt1(mean, std):
    return f"{mean:.1f} ± {std:.1f}"


def smooth5(y):
    y = np.asarray(y, dtype=np.float64)
    out = np.full(y.shape, np.nan)
    c = np.cumsum(y)
    out[4:] = (c[4:] - np.concatenate([[0.0], c[:-5]])) / 5.0
    return out


def _fonts():
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 10,
    })


def plot_fig1(olla_eval):
    _fonts()
    colors = {"alea": "C1", "epist": "C0", "epsilon": "0.45"}
    names = {"alea": "aleatoric", "epist": "epistemic", "epsilon": "eps-greedy"}
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    modes = ["alea", "epist", "epsilon"]
    for mode in modes:
        rows = []
        for seed in TRAIN_SEEDS:
            z = np.load(OUT / f"{mode}_seed{seed}_log.npz")
            rows.append(smooth5(z["train_return"]))
        stack = np.stack(rows)
        x = np.arange(1, stack.shape[1] + 1)
        mu = np.nanmean(stack, axis=0)
        sd = np.nanstd(stack, axis=0, ddof=1)
        ok = np.isfinite(mu)
        ax.fill_between(x[ok], (mu - sd)[ok], (mu + sd)[ok], color=colors[mode], alpha=0.18, lw=0)
        ax.plot(x[ok], mu[ok], color=colors[mode], lw=1.8, label=names[mode])
    ax.axvline(58, color="0.35", ls=":", lw=1.0, label="ε reaches 0.01")
    ax.axhline(olla_eval, color="C3", ls="--", lw=1.0, label=f"OLLA eval {olla_eval:.1f}")
    ax.set_xlabel("episode")
    ax.set_ylabel("episode return")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="lower right", ncol=2, frameon=True, framealpha=0.9)
    fig.tight_layout()
    path = OUT / "fig1_returns.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}", flush=True)


def sigmas_of(agent, state):
    _mu, sig = agent._forward_all(state)
    return sig.mean(dim=0).detach().cpu().numpy()


def rollout(env, choose, seed, agent=None):
    seed_phy(seed)
    state, info = env.reset(seed=seed)
    mcs, sig_rows, acks, retx = [], [], [], []
    overflow = 0
    discard = 0
    episode_return = 0.0
    done = False
    while not done:
        action = int(choose(state, info))
        if agent is not None:
            sig_rows.append(sigmas_of(agent, state))
        state, reward, terminated, truncated, info = env.step(action)
        episode_return += float(reward)
        done = terminated or truncated
        out = info["outcome"]
        overflow += int(out.get("n_overflow", 0))
        discard += int(out.get("discard_bits", 0))
        if out.get("idle"):
            continue
        mcs.append(int(out["mcs_used"]))
        acks.append(int(out["ack"]))
        retx.append(int(out["num_retx"]))
    mcs = np.asarray(mcs, dtype=np.int64)
    acks = np.asarray(acks, dtype=np.float64)
    retx = np.asarray(retx, dtype=np.float64)
    stats = {
        "bler": float(1.0 - acks.mean()) if acks.size else float("nan"),
        "retx": float(retx.mean()) if retx.size else float("nan"),
        "overflow_bits": float(overflow),
        "discard_bits": float(discard),
        "final_queue": float(info["queue"]),
        "episode_return": float(episode_return),
        "n_tb": int(mcs.size),
    }
    sig = np.stack(sig_rows) if sig_rows else None
    return stats, mcs, sig


def plot_fig2(mcs_rate, bin_sigma):
    _fonts()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    width = 2.3
    series = (
        ("alea", "C1", -1, "aleatoric"),
        ("epist", "C0", 0, "epistemic"),
        ("epsilon", "0.55", 1, "eps-greedy, ε=0.01"),
    )
    ymax = 0.0
    for i, (name, lo, hi) in enumerate(BINS):
        center = 0.5 * (lo + hi)
        sl = slice(lo - MCS_MIN, hi - MCS_MIN + 1)
        for mode, color, shift, label in series:
            mass = float(np.sum(mcs_rate[mode][sl]))
            ymax = max(ymax, mass)
            ax.bar(
                center + shift * width,
                mass,
                width=width * 0.92,
                color=color,
                alpha=0.9,
                label=label if i == 0 else None,
            )
    ax.set_ylim(0.0, ymax * 1.28)
    ax.set_ylabel("selection mass")
    ax.set_xlim(MCS_MIN - 0.5, MCS_MAX + 0.5)
    ax.set_xticks([7, 16, 24.5])
    ax.set_xticklabels(
        [
            f"low 3-11\n(aleatoric σ̄={bin_sigma[0]:.2f})",
            f"mid 12-20\n(aleatoric σ̄={bin_sigma[1]:.2f})",
            f"high 21-28\n(aleatoric σ̄={bin_sigma[2]:.2f})",
        ]
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    path = OUT / "fig2_mcs_sigma.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}", flush=True)


def spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rx = x.argsort().argsort().astype(np.float64)
    ry = y.argsort().argsort().astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom)


def main():
    global OUT
    import argparse
    import json

    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=OUT)
    args = p.parse_args()
    OUT = args.out_dir if args.out_dir.is_absolute() else HERE / args.out_dir
    np.random.seed(0)
    torch.manual_seed(0)
    cfg = load_config()
    phy = PHYAbstraction()
    env = DownlinkLAEnv.from_config(cfg, phy_abs=phy)
    olla = make_baseline_policy(
        "olla",
        env,
        bler_target=float(cfg["bler_target"]),
        olla_step_up_db=cfg.get("olla_step_up_db"),
    )
    behavior_modes = ["alea", "epist"]
    if (OUT / "epsilon_seed0.pt").is_file():
        behavior_modes.append("epsilon")
    agents = {
        mode: {s: load_mode(OUT / f"{mode}_seed{s}.pt") for s in TRAIN_SEEDS}
        for mode in behavior_modes
    }
    for seed, agent in agents.get("epsilon", {}).items():
        agent.total_steps = agent.epsilon_decay_steps

    raw_rows = []
    per_seed = {mode: [] for mode in behavior_modes}
    hist = {mode: [] for mode in ("alea", "epist", "epsilon", "OLLA")}
    count_book = {mode: [] for mode in behavior_modes}
    sigma_acc = []

    for train_seed in TRAIN_SEEDS:
        for mode in behavior_modes:
            agent = agents[mode][train_seed]
            seed_stats = []
            counts = np.zeros(MCS_MAX + 1, dtype=np.float64)
            n_tb = 0
            sig_sum = np.zeros(env.action_space.n, dtype=np.float64)
            n_sig = 0
            for eval_seed in EVAL_SEEDS:
                stats, mcs, sig = rollout(
                    env,
                    lambda st, info, a=agent: a.select_action(st, greedy=False),
                    eval_seed,
                    agent=agent if mode == "alea" else None,
                )
                seed_stats.append(stats)
                raw_rows.append((train_seed, mode, eval_seed, stats))
                counts += np.bincount(mcs, minlength=MCS_MAX + 1)
                n_tb += mcs.size
                if sig is not None:
                    sig_sum += sig.sum(axis=0)
                    n_sig += sig.shape[0]
                print(
                    f"behavior {mode} train {train_seed} eval {eval_seed} "
                    f"BLER={stats['bler']:.3f} retx={stats['retx']:.2f}",
                    flush=True,
                )
            per_seed[mode].append({k: float(np.mean([s[k] for s in seed_stats])) for k in seed_stats[0]})
            hist[mode].append(counts / max(n_tb, 1))
            count_book[mode].append((counts.copy(), int(n_tb)))
            if mode == "alea" and n_sig:
                sigma_acc.append(sig_sum / n_sig)

    olla_stats = []
    counts = np.zeros(MCS_MAX + 1, dtype=np.float64)
    n_tb = 0
    for eval_seed in EVAL_SEEDS:
        olla.reset()
        stats, mcs, _sig = rollout(env, olla, eval_seed)
        olla_stats.append(stats)
        raw_rows.append(("none", "OLLA", eval_seed, stats))
        counts += np.bincount(mcs, minlength=MCS_MAX + 1)
        n_tb += mcs.size
        print(f"OLLA eval {eval_seed} BLER={stats['bler']:.3f} retx={stats['retx']:.2f}", flush=True)
    hist["OLLA"] = counts / max(n_tb, 1)
    olla_eval = float(np.mean([row["episode_return"] for row in olla_stats]))
    plot_fig1(olla_eval)

    xs = np.arange(MCS_MIN, MCS_MAX + 1)
    mcs_rate = {
        mode: np.mean([h[xs] for h in hist[mode]], axis=0)
        for mode in behavior_modes
    }
    sigma_action = np.mean(np.stack(sigma_acc), axis=0)
    sigma_profile = sigma_action[xs - MCS_MIN]
    pooled_rate = {}
    bin_lines = [
        "# MCS bin counts. Pooled over 3 training seeds and eval seeds 205-209.",
        "Epsilon rows use epsilon=0.01. Alea rows sample N(mean mu, (mean sigma)^2).",
        "mean_sigma is the unweighted mean of the per-MCS alea-ensemble sigma in that bin.",
        "That sigma is averaged on states visited by the alea behavior.",
        "",
        "| policy | bin | MCS | count | n_tb | rate | mean_sigma |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for mode in ("alea", "epsilon", "epist"):
        if mode not in count_book or not count_book[mode]:
            continue
        total_c = np.sum([c for c, _n in count_book[mode]], axis=0)
        total_n = int(sum(n for _c, n in count_book[mode]))
        pooled_rate[mode] = total_c[xs] / max(total_n, 1)
        for name, lo, hi in BINS:
            sl = slice(lo, hi + 1)
            count = int(total_c[sl].sum())
            sig = float(sigma_profile[lo - MCS_MIN : hi - MCS_MIN + 1].mean())
            bin_lines.append(
                f"| {mode} | {name} | {lo}-{hi} | {count} | {total_n} | "
                f"{count / max(total_n, 1):.6f} | {sig:.4f} |"
            )
    if "epsilon" not in pooled_rate:
        raise SystemExit("epsilon checkpoints are required for the baseline bars")
    for mode in ("alea", "epist", "epsilon"):
        mcs_rate[mode] = pooled_rate[mode]
    bin_sigma = [
        float(sigma_profile[lo - MCS_MIN : hi - MCS_MIN + 1].mean())
        for _name, lo, hi in BINS
    ]
    rho_s = spearman(mcs_rate["alea"], sigma_profile)
    (OUT / "mcs_bins.md").write_text("\n".join(bin_lines) + "\n", encoding="utf-8")
    per_mcs = ["mcs,alea_rate,alea_sigma"]
    for i, m in enumerate(xs):
        per_mcs.append(f"{int(m)},{mcs_rate['alea'][i]:.8f},{sigma_profile[i]:.6f}")
    (OUT / "mcs_alea_sigma.csv").write_text("\n".join(per_mcs) + "\n", encoding="utf-8")
    plot_fig2(mcs_rate, bin_sigma)

    show = ["episode_return", "bler", "discard_bits", "overflow_bits"]
    labels = {
        "episode_return": "episode return",
        "bler": "first-tx BLER",
        "retx": "mean retransmissions",
        "final_queue": "final queue [bits]",
        "discard_bits": "HARQ discard [bits]",
        "overflow_bits": "overflow [bits]",
    }

    def cell(mode_rows, k, olla=False):
        mu, sd = mean_std([row[k] for row in mode_rows])
        return (fmt(mu, sd) if k in ("bler", "retx") else fmt1(mu, sd)), mu, sd

    alea_mass = []
    epist_mass = []
    for _name, lo, hi in BINS:
        sl = slice(lo - MCS_MIN, hi - MCS_MIN + 1)
        alea_mass.append(float(mcs_rate["alea"][sl].sum()))
        epist_mass.append(float(mcs_rate["epist"][sl].sum()))
    greedy_path = OUT / "greedy_means.json"
    if greedy_path.is_file():
        greedy_raw = json.loads(greedy_path.read_text(encoding="utf-8"))
        greedy_vals = {
            mode: [greedy_raw[str(s)][mode] for s in TRAIN_SEEDS]
            for mode in ("alea", "epist")
            if str(TRAIN_SEEDS[0]) in greedy_raw and mode in greedy_raw[str(TRAIN_SEEDS[0])]
        }
    else:
        greedy_vals = GREEDY
    g_alea, g_alea_sd = mean_std(greedy_vals["alea"])
    g_epist, g_epist_sd = mean_std(greedy_vals["epist"])
    retx_bits = []
    for mode in behavior_modes:
        mu, sd = mean_std([row["retx"] for row in per_seed[mode]])
        qmu, qsd = mean_std([row["final_queue"] for row in per_seed[mode]])
        retx_bits.append((mu, sd, qmu, qsd))
    o_retx, o_retx_sd = mean_std([row["retx"] for row in olla_stats])
    o_q, o_q_sd = mean_std([row["final_queue"] for row in olla_stats])
    olla_by_eval = {
        eval_seed: stats["episode_return"]
        for train_seed, mode, eval_seed, stats in raw_rows
        if mode == "OLLA"
    }
    paired_lines = []
    for mode in behavior_modes:
        seed_gaps = []
        seed_bits = []
        for i, train_seed in enumerate(TRAIN_SEEDS):
            # per_seed is the mean over eval seeds; rebuild paired from raw_rows
            diffs = []
            disc = []
            for tr, md, ev, st in raw_rows:
                if tr == train_seed and md == mode:
                    diffs.append(st["episode_return"] - olla_by_eval[ev])
                    disc.append(st["discard_bits"])
            seed_gaps.append(float(np.mean(diffs)))
            seed_bits.append(float(np.mean(disc)))
        gmu, gsd = mean_std(seed_gaps)
        paired_lines.append(
            f"{mode}: train-seed paired gaps {', '.join(f'{v:+.1f}' for v in seed_gaps)}; "
            f"mean {gmu:+.1f} ± {gsd:.1f}. "
            f"HARQ discard by train seed {', '.join(f'{v:.1f}' for v in seed_bits)}."
        )

    lines = [
        "# gamma=0.9, 120 episodes",
        "",
        "Training seeds are 0, 1000, and 2000 (n=3). Each training seed is averaged",
        "over eval seeds 205-209 before the mean and standard deviation across training",
        "seeds are taken. ± values are the sample standard deviation (ddof=1), not the",
        "standard error. n=3 is too small for a significance test.",
        "This file, Fig. 1, and Fig. 2 are one run of report_des32.py.",
        "",
        "Paired return is the policy minus OLLA on the same eval seed,",
        "averaged over seeds 205-209 within a training seed, then mean ± standard",
        "deviation across the 3 training seeds. OLLA's own ± is across eval seeds",
        "and is not the same kind of number.",
        "",
        *paired_lines,
        "",
        f"Spearman correlation of alea per-MCS selection rate and alea sigma: {rho_s:.3f}.",
        f"Bin mean sigma from the alea model: low {bin_sigma[0]:.2f}, "
        f"mid {bin_sigma[1]:.2f}, high {bin_sigma[2]:.2f}.",
        "",
        "Fig. 1 caption: training episode return, 5-episode trailing mean.",
        "Band is ±1 standard deviation across 3 training seeds.",
        "The vertical line is where epsilon reaches 0.01 (30000 decisions,",
        "about 516 decisions per episode, episode 58).",
        f"The horizontal line is the OLLA evaluation mean {olla_eval:.1f}",
        "on seeds 205-209, not a training curve.",
        "",
        "Fig. 2 caption: selection mass on MCS 3-11, 12-20, and 21-28.",
        "Tick labels give the unweighted mean of per-MCS sigma from the alea",
        "ensemble, averaged on states the alea behavior visited.",
        "Epsilon bars use epsilon=0.01.",
        "",
        "Episode return is the sum of delivered bits/RE over the 1000-slot episode.",
        "Fig. 1 plots that quantity during training. The table column is the same",
        "quantity on eval seeds 205-209, then summarized over the 3 training seeds.",
        "Overflow and HARQ discard penalties are zero, so return equals delivered",
        "spectral efficiency; the loss columns are the bits that were not delivered.",
        "",
        "Behavior rows use the training action rule. OLLA has no training seed;",
        "its ± value is the standard deviation across the 5 eval seeds.",
        "",
        "Epsilon-greedy uses the same ensemble, gamma, episodes, and training seeds.",
        "During training, epsilon decays from 1 to 0.01 over 30000 decisions.",
        "One episode is about 516 decisions, so this ends near episode 58,",
        "not at episode 30. The table row evaluates the saved agent at",
        "epsilon=0.01. Alea and epist rows keep their own stationary rules",
        "(Gaussian sample, or one member's mean). They do not use epsilon.",
        "",
        "Epistemic sampling argmaxes one member's mean mu and ignores sigma.",
        "Its MCS mass on 3-11 / 12-20 / 21-28 is",
        f"{epist_mass[0]:.2f} / {epist_mass[1]:.2f} / {epist_mass[2]:.2f}.",
        "It does select MCS above 12; the mode is simply lower because high MCS",
        "has a lower mean return under r - rho * tau, not because index 13 is masked.",
        "",
        "Fig. 2 shows only the aleatoric behavior. Its selection mass is",
        f"{alea_mass[0]:.2f} / {alea_mass[1]:.2f} / {alea_mass[2]:.2f}",
        "on MCS 3-11 / 12-20 / 21-28, and sigma rises toward the high bin.",
        "Epistemic behavior, not drawn, puts",
        f"{epist_mass[0]:.2f} / {epist_mass[1]:.2f} / {epist_mass[2]:.2f}",
        "on those same bins, so most of its mass is in the low bin.",
        "",
        "Greedy return is the mean mu policy, already averaged over eval seeds 205-209.",
        f"Across 3 training seeds it is alea {g_alea:.1f} ± {g_alea_sd:.1f},",
        f"epist {g_epist:.1f} ± {g_epist_sd:.1f}.",
        "Seed 0 favors epist; seeds 1000 and 2000 favor alea.",
        "",
        "Mean retransmissions are",
        f"alea {retx_bits[0][0]:.3f} ± {retx_bits[0][1]:.3f},",
        f"epist {retx_bits[1][0]:.3f} ± {retx_bits[1][1]:.3f},",
        f"OLLA {o_retx:.3f} ± {o_retx_sd:.3f}.",
        "Final queue bits are",
        f"alea {retx_bits[0][2]:.0f} ± {retx_bits[0][3]:.0f},",
        f"epist {retx_bits[1][2]:.0f} ± {retx_bits[1][3]:.0f},",
        f"OLLA {o_q:.0f} ± {o_q_sd:.0f}.",
        "",
        "| policy | n | " + " | ".join(labels[k] for k in show) + " |",
        "| --- | --- | " + " | ".join("---" for _ in show) + " |",
    ]
    keys = show
    for mode in behavior_modes:
        cells = []
        for k in keys:
            mu, sd = mean_std([row[k] for row in per_seed[mode]])
            cells.append(fmt(mu, sd) if k == "bler" or k == "retx" else fmt1(mu, sd))
        lines.append(f"| {mode} behavior | 3 train seeds | " + " | ".join(cells) + " |")
    cells = []
    for k in keys:
        mu, sd = mean_std([row[k] for row in olla_stats])
        cells.append(fmt(mu, sd) if k in ("bler", "retx") else fmt1(mu, sd))
    lines.append("| OLLA | 5 eval seeds | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Per training seed, after averaging eval seeds 205-209",
        "",
        "| train seed | policy | " + " | ".join(labels[k] for k in keys) + " |",
        "| --- | --- | " + " | ".join("---" for _ in keys) + " |",
    ]
    for i, train_seed in enumerate(TRAIN_SEEDS):
        for mode in behavior_modes:
            row = per_seed[mode][i]
            cells = [
                f"{row[k]:.3f}" if k in ("bler", "retx") else f"{row[k]:.1f}" for k in keys
            ]
            lines.append(f"| {train_seed} | {mode} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Every eval episode",
        "",
        "| train seed | policy | eval seed | " + " | ".join(labels[k] for k in keys) + " |",
        "| --- | --- | --- | " + " | ".join("---" for _ in keys) + " |",
    ]
    for train_seed, mode, eval_seed, stats in raw_rows:
        cells = [
            f"{stats[k]:.3f}" if k in ("bler", "retx") else f"{stats[k]:.1f}" for k in keys
        ]
        lines.append(f"| {train_seed} | {mode} | {eval_seed} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    path = OUT / "table.md"
    path.write_text(text, encoding="utf-8")
    print(f"saved {path}", flush=True)
    print("\n".join(lines[: 14 + 3]), flush=True)


if __name__ == "__main__":
    main()
