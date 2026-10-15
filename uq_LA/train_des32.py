# Desideratum 3.2: same ensemble, two action rules, 80 episodes.
# alea: y ~ N(mean mu, (mean sigma)^2), then argmax.
# epist: one member, then argmax of that member's mu.
# Greedy eval always uses the mean mu. No epsilon-greedy.

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
from train_uq import rollout_metrics
from uq_ensemble import EnsembleDQNAgent

HERE = Path(__file__).resolve().parent
EPISODES = 80
PROBE_EVERY = 20
PROBE_SEED = 205
EVAL_SEEDS = [205, 206, 207]
MODES = ("alea", "epist")


def make_agent(env, cfg, seed):
    uq = cfg.get("uq_dqn") or {}
    ens = cfg.get("ensemble") or {}
    return EnsembleDQNAgent(
        int(np.prod(env.state_space.shape)),
        env.action_space.n,
        n_models=int(ens.get("n_models", 5)),
        hidden=int(ens.get("hidden", uq.get("hidden", 256))),
        gamma=float(uq.get("gamma", 0.7)),
        lr=float(uq.get("lr", 1e-3)),
        buffer_size=int(uq.get("buffer_size", 100_000)),
        batch_size=int(uq.get("batch_size", 64)),
        target_sync=int(uq.get("target_sync", 200)),
        rho=0.0,
        seed=seed,
    )


def train_one(env, cfg, mode, seed, rho_ema, episodes, probe_every):
    np.random.seed(seed)
    torch.manual_seed(seed)
    agent = make_agent(env, cfg, seed)
    agent.action_mode = mode
    log = {k: [] for k in ("episode", "train_return", "greedy_return", "rho")}
    print(f"\n=== {mode} seed {seed} {episodes} ep ===", flush=True)
    for ep in range(episodes):
        m, _ = rollout_metrics(env, agent, seed=seed + ep, train=True)
        rate = m["return_per_slot"]
        agent.rho = rate if ep == 0 else (1.0 - rho_ema) * agent.rho + rho_ema * rate
        log["episode"].append(ep + 1)
        log["train_return"].append(m["ret"])
        greedy = np.nan
        if (ep + 1) % probe_every == 0 or ep + 1 == episodes:
            seed_phy(PROBE_SEED)
            g, _ = rollout_metrics(env, agent, seed=PROBE_SEED, train=False, greedy=True)
            greedy = g["ret"]
            print(
                f"  ep {ep + 1:4d} train={m['ret']:.1f} greedy205={greedy:.1f} "
                f"rho={agent.rho:.3f} nll={agent.last_nll:.3f}",
                flush=True,
            )
        log["greedy_return"].append(greedy)
        log["rho"].append(agent.rho)
    return agent, log


def eval_greedy(env, agent, olla, seeds):
    rows = []
    olla_rows = []
    for s in seeds:
        seed_phy(s)
        m, _ = rollout_metrics(env, agent, seed=s, train=False, greedy=True)
        seed_phy(s)
        olla.reset()
        state, info = env.reset(seed=s)
        total = 0.0
        done = False
        while not done:
            action = olla(state, info)
            state, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            done = terminated or truncated
        rows.append(m["ret"])
        olla_rows.append(total)
        print(f"  seed {s} {agent.action_mode} {m['ret']:.1f} OLLA {total:.1f}", flush=True)
    return np.asarray(rows), np.asarray(olla_rows)


def plot(logs, evals, olla_mean, path, gamma):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    colors = {"alea": "C1", "epist": "C0"}
    for mode, log in logs.items():
        y = np.asarray(log["train_return"], dtype=np.float64)
        ep = np.asarray(log["episode"])
        kernel = np.ones(10) / 10.0
        smooth = np.convolve(y, kernel, mode="valid")
        axes[0].plot(ep, y, color=colors[mode], alpha=0.25, lw=0.8)
        axes[0].plot(ep[9:], smooth, color=colors[mode], lw=1.6, label=mode)
        g_ep = ep[np.isfinite(log["greedy_return"])]
        g_y = np.asarray(log["greedy_return"], dtype=np.float64)
        g_y = g_y[np.isfinite(g_y)]
        axes[1].plot(g_ep, g_y, "o-", color=colors[mode], label=mode)
    axes[0].set_title("training return (behavior policy)")
    axes[0].set_xlabel("episode")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].axhline(olla_mean, color="C3", ls="--", lw=1.0, label=f"OLLA {olla_mean:.0f}")
    axes[1].set_title(f"greedy return, seed {PROBE_SEED}")
    axes[1].set_xlabel("episode")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(
        f"Desideratum 3.2, gamma={gamma:g}: sampling-aleatoric vs sampling-epistemic",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    labels = ["alea", "epist", "OLLA"]
    means = [evals["alea"][0].mean(), evals["epist"][0].mean(), olla_mean]
    ses = [
        evals["alea"][0].std(ddof=1) / np.sqrt(len(evals["alea"][0])),
        evals["epist"][0].std(ddof=1) / np.sqrt(len(evals["epist"][0])),
        evals["alea"][1].std(ddof=1) / np.sqrt(len(evals["alea"][1])),
    ]
    ax.bar(labels, means, color=["C1", "C0", "C3"], yerr=ses, capsize=4)
    ax.set_ylabel("greedy episode return")
    ax.set_title(f"seeds {EVAL_SEEDS[0]}-{EVAL_SEEDS[-1]}")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path.with_name("eval_bars.png"), dpi=150)
    plt.close(fig)


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--episodes", type=int, default=EPISODES)
    p.add_argument("--train-seeds", type=int, nargs="+", default=None)
    p.add_argument("--eval-seeds", type=int, nargs="+", default=None)
    p.add_argument("--probe-every", type=int, default=PROBE_EVERY)
    p.add_argument("--modes", nargs="+", default=list(MODES))
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    with (HERE / "configs" / "downlink_la.yaml").open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if args.gamma is not None:
        cfg.setdefault("uq_dqn", {})["gamma"] = float(args.gamma)
    gamma = float((cfg.get("uq_dqn") or {}).get("gamma", 0.7))
    episodes = int(args.episodes)
    modes = tuple(args.modes)
    train_seeds = [int(s) for s in (args.train_seeds or [cfg.get("seed", 0)])]
    eval_seeds = [int(s) for s in (args.eval_seeds or EVAL_SEEDS)]
    rho_ema = float((cfg.get("uq_dqn") or {}).get("rho_ema", 0.1))
    phy = PHYAbstraction()
    env = DownlinkLAEnv.from_config(cfg, phy_abs=phy)
    out = args.out_dir or (HERE / "outputs" / "des32")
    if not out.is_absolute():
        out = HERE / out
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"gamma={gamma} episodes={episodes} train_seeds={train_seeds} "
        f"eval_seeds={eval_seeds} modes={modes} out={out}",
        flush=True,
    )

    olla = make_baseline_policy(
        "olla",
        env,
        bler_target=float(cfg["bler_target"]),
        olla_step_up_db=cfg.get("olla_step_up_db"),
    )
    by_seed = {}
    for train_seed in train_seeds:
        logs = {}
        agents = {}
        for mode in modes:
            agents[mode], logs[mode] = train_one(
                env, cfg, mode, train_seed, rho_ema, episodes, int(args.probe_every)
            )
            torch.save(
                {
                    **agents[mode].state_dict(),
                    "action_mode": mode,
                    "gamma": agents[mode].gamma,
                    "rho": agents[mode].rho,
                    "episodes": episodes,
                    "train_seed": train_seed,
                },
                out / f"{mode}_seed{train_seed}.pt",
            )
            np.savez(
                out / f"{mode}_seed{train_seed}_log.npz",
                **{k: np.asarray(v) for k, v in logs[mode].items()},
            )
        evals = {}
        for mode in modes:
            print(f"\n=== greedy eval {mode} train_seed {train_seed} ===", flush=True)
            evals[mode] = eval_greedy(env, agents[mode], olla, eval_seeds)
        by_seed[train_seed] = {"logs": logs, "evals": evals}
        del agents

    def mean_se(vals):
        vals = np.asarray(vals, dtype=np.float64)
        se = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else float("nan")
        return float(vals.mean()), se

    if set(modes) != {"alea", "epist"}:
        lines = [
            f"# epsilon-greedy ensemble. gamma={gamma}, {episodes} episodes.",
            f"Train seeds {train_seeds}. Eval seeds {eval_seeds}.",
            "Action rule: epsilon-greedy on the mean mu. Same ensemble and NLL as alea/epist.",
            "",
            "| train seed | mode | last 10 train return | greedy mean |",
            "| --- | --- | ---: | ---: |",
        ]
        for train_seed, pack in by_seed.items():
            for mode in modes:
                last = float(np.mean(pack["logs"][mode]["train_return"][-10:]))
                gmean = float(pack["evals"][mode][0].mean())
                lines.append(f"| {train_seed} | {mode} | {last:.1f} | {gmean:.1f} |")
        text = "\n".join(lines) + "\n"
        (out / "train_summary.md").write_text(text, encoding="utf-8")
        greedy = {
            str(train_seed): {
                mode: float(pack["evals"][mode][0].mean()) for mode in modes
            }
            for train_seed, pack in by_seed.items()
        }
        (out / "greedy_means.json").write_text(
            __import__("json").dumps(greedy), encoding="utf-8"
        )
        print(text, flush=True)
        print(f"wrote {out / 'train_summary.md'}", flush=True)
        return

    last_gap, greedy_gap = [], []
    lines = [
        f"# Desideratum 3.2. gamma={gamma}, {episodes} episodes.",
        f"Train seeds {train_seeds}. Each seed uses env seeds "
        f"[seed, seed+{episodes}).",
        "alea: one N(mean mu, (mean sigma)^2) sample per action, then argmax.",
        "epist: one ensemble member, then argmax of its mu.",
        f"Greedy eval uses the mean mu on seeds {eval_seeds}.",
        "",
        "| train seed | alea last10 | epist last10 | epist-alea | alea greedy | epist greedy | greedy epist-alea |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for train_seed, pack in by_seed.items():
        last = {
            mode: float(np.mean(pack["logs"][mode]["train_return"][-10:])) for mode in modes
        }
        gmean = {mode: float(pack["evals"][mode][0].mean()) for mode in modes}
        last_gap.append(last["epist"] - last["alea"])
        greedy_gap.append(gmean["epist"] - gmean["alea"])
        lines.append(
            f"| {train_seed} | {last['alea']:.1f} | {last['epist']:.1f} | "
            f"{last_gap[-1]:+.1f} | {gmean['alea']:.1f} | {gmean['epist']:.1f} | "
            f"{greedy_gap[-1]:+.1f} |"
        )
    lg_mean, lg_se = mean_se(last_gap)
    gg_mean, gg_se = mean_se(greedy_gap)
    lines += [
        "",
        f"**epist - alea, last 10 training episodes:** {lg_mean:+.1f} ± {lg_se:.1f} "
        f"(SE over {len(train_seeds)} train seeds)",
        "",
        f"**epist - alea, greedy mean on eval seeds:** {gg_mean:+.1f} ± {gg_se:.1f} "
        f"(SE over {len(train_seeds)} train seeds)",
        "",
    ]
    text = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(text, encoding="utf-8")
    print(text, flush=True)

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    colors = {"alea": "C1", "epist": "C0"}
    for mode in MODES:
        curves = [
            np.asarray(by_seed[s]["logs"][mode]["train_return"], dtype=np.float64)
            for s in train_seeds
        ]
        stack = np.stack(curves)
        mean = stack.mean(axis=0)
        ep = np.arange(1, episodes + 1)
        for row in stack:
            ax.plot(ep, row, color=colors[mode], alpha=0.25, lw=0.8)
        ax.plot(ep, mean, color=colors[mode], lw=1.8, label=f"{mode} mean")
    ax.set_xlabel("episode")
    ax.set_ylabel("training return")
    ax.set_title(f"gamma={gamma:g}, train seeds {train_seeds}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "curves.png", dpi=150)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
