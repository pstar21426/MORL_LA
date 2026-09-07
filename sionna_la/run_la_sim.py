# ILLA vs OLLA rollout on  DownlinkLAEnv

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from sionna.sys import PHYAbstraction

from la_env import DownlinkLAEnv, seed_phy
from policies import EpsilonGreedyPolicy, make_baseline_policy


def load_config(path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def rollout(env, policy, seed):
    state, info = env.reset(seed=seed)
    policy.reset()

    log = {k: [] for k in (
        "state", "action", "reward", "next_state", "done",
        "mcs_used", "ack", "tb_success", "dropped", "truncated_mid_tb",
        "num_slots", "num_retx", "delta_tau",
        "sinr_true_db", "sinr_hat_db", "cqi_index", "cqi_norm",
        "tbler", "tbler_last", "decoded_bits",
    )}

    done = False
    while not done:
        action = policy(state, info)
        delta_tau = float(info.get("delta_tau", 0.0))
        sinr_hat = float(info["sinr_hat_db"])
        cqi_index = int(info["cqi_index"])
        cqi_norm = float(state[0])

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        out = info["outcome"]

        log["state"].append(state)
        log["action"].append(action)
        log["reward"].append(reward)
        log["next_state"].append(next_state)
        log["done"].append(done)
        log["mcs_used"].append(out["mcs_used"])
        log["ack"].append(out["ack"])
        log["tb_success"].append(out["tb_success"])
        log["dropped"].append(out["dropped"])
        log["truncated_mid_tb"].append(out["truncated_mid_tb"])
        log["num_slots"].append(out["num_slots"])
        log["num_retx"].append(out["num_retx"])
        log["delta_tau"].append(delta_tau)
        log["sinr_true_db"].append(out["sinr_true_db"])
        log["sinr_hat_db"].append(sinr_hat)
        log["cqi_index"].append(cqi_index)
        log["cqi_norm"].append(cqi_norm)
        log["tbler"].append(out["tbler"])
        log["tbler_last"].append(out["tbler_last"])
        log["decoded_bits"].append(out["decoded_bits"])
        state = next_state

    out = {k: np.asarray(v) for k, v in log.items()}
    out["sinr_true_trace"] = np.asarray(env._sinr_true_db, dtype=np.float64)
    out["sinr_hat_trace"] = np.asarray(env._sinr_fb_db, dtype=np.float64)
    out["cqi_delay_slots"] = np.int64(env._delay_used)
    return out


def metrics_from_rollout(res, num_slots):
    n = len(res["ack"])
    finished = ~np.asarray(res["truncated_mid_tb"], dtype=bool)
    n_fin = int(finished.sum())
    tb_fail = (
        float(1.0 - res["tb_success"][finished].mean()) if n_fin else 0.0
    )
    return {
        "return": float(res["reward"].sum()),
        "throughput": float(res["reward"].sum() / num_slots),
        "tbs": n,
        "first_tx_bler": float(1.0 - res["ack"].mean()),
        "tb_fail": tb_fail,
        "mean_mcs": float(res["mcs_used"].mean()),
        "mean_slots_tb": float(res["num_slots"].mean()),
        "drops": int(res["dropped"].sum()),
    }


def summarize(name, res, bler_target, num_slots):
    m = metrics_from_rollout(res, num_slots)
    print(
        f"[{name.upper()}] "
        f"return={m['return']:.1f} | "
        f"throughput={m['throughput']:.4f} | "
        f"TBs={m['tbs']} | "
        f"first-tx BLER={m['first_tx_bler']:.3f} (target {bler_target}) | "
        f"TB fail={m['tb_fail']:.3f} | "
        f"mean MCS={m['mean_mcs']:.1f} | "
        f"mean slots/TB={m['mean_slots_tb']:.2f} | "
        f"drops={m['drops']}"
    )
    return m


def aggregate_metrics(rows):
    keys = [
        "return", "throughput", "tbs", "first_tx_bler", "tb_fail",
        "mean_mcs", "mean_slots_tb", "drops",
    ]
    out = {}
    for k in keys:
        vals = np.asarray([r[k] for r in rows], dtype=np.float64)
        out[f"{k}_mean"] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std())
    return out


def print_mean_row(name, agg, bler_target):
    print(
        f"[{name.upper():>5}] "
        f"return={agg['return_mean']:7.1f} ± {agg['return_std']:.1f} | "
        f"throughput={agg['throughput_mean']:.4f} ± {agg['throughput_std']:.4f} | "
        f"TBs={agg['tbs_mean']:.0f} | "
        f"1st-tx BLER={agg['first_tx_bler_mean']:.3f} ± {agg['first_tx_bler_std']:.3f} "
        f"(tgt {bler_target}) | "
        f"slots/TB={agg['mean_slots_tb_mean']:.2f} | "
        f"drops={agg['drops_mean']:.1f}"
    )


def _rolling(x, window):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return x
    w = max(1, min(int(window), len(x)))
    if w == 1:
        return x
    c = np.cumsum(x, dtype=np.float64)
    out = np.empty_like(x)
    out[: w - 1] = c[: w - 1] / np.arange(1, w)
    out[w - 1 :] = (c[w - 1 :] - np.concatenate(([0.0], c[: len(x) - w]))) / w
    return out


def _slot_cum_return(rewards, slots_per_tb, total_slots):
    y = np.zeros(int(total_slots), dtype=np.float64)
    t = 0
    cum = 0.0
    for r, ns in zip(rewards, slots_per_tb):
        ns = max(int(ns), 1)
        cum += float(r)
        end = min(t + ns, total_slots)
        y[t:end] = cum
        t = end
        if t >= total_slots:
            break
    if t < total_slots:
        y[t:] = cum
    return y


def _slot_hold(values, slots_per_tb, total_slots):
    y = np.empty(int(total_slots), dtype=np.float64)
    t = 0
    last = float(values[0]) if len(values) else 0.0
    for v, ns in zip(values, slots_per_tb):
        ns = max(int(ns), 1)
        last = float(v)
        end = min(t + ns, total_slots)
        y[t:end] = last
        t = end
        if t >= total_slots:
            break
    if t < total_slots:
        y[t:] = last
    return y


def _flush_xlim(ax, x_max):
    ax.set_xlim(0, x_max)
    ax.margins(x=0)


def _tb_starts(slots_per_tb, total_slots):
    starts = []
    t = 0
    for ns in slots_per_tb:
        if t >= total_slots:
            break
        starts.append(t)
        t += max(int(ns), 1)
    return np.asarray(starts, dtype=int)


def _running_mean(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return x
    return np.cumsum(x) / np.arange(1, len(x) + 1)


def plot_results(results, bler_target, out_path, num_slots, mcs_window=50):
    fig, axs = plt.subplots(5, 1, figsize=(9, 13.5), sharex=True)

    ref = next(iter(results.values()))
    slots = np.arange(num_slots)
    x_max = num_slots - 1
    delay = int(np.asarray(ref.get("cqi_delay_slots", -1)).reshape(-1)[0])

    sinr_true = ref.get("sinr_true_trace")
    sinr_hat = ref.get("sinr_hat_trace")
    if sinr_true is None:
        sinr_true = _slot_hold(ref["sinr_true_db"], ref["num_slots"], num_slots)
    if sinr_hat is None:
        sinr_hat = _slot_hold(ref["sinr_hat_db"], ref["num_slots"], num_slots)
    axs[0].plot(slots, sinr_true[:num_slots], label="true SINR", color="0.35", lw=0.9)
    axs[0].plot(
        slots,
        sinr_hat[:num_slots],
        ":",
        label="SINR hat (pre-CQI)",
        color="C1",
        alpha=0.9,
    )
    axs[0].set_ylabel("SINR [dB]")
    delay_txt = f"cqi_delay={delay} slots" if delay >= 0 else "cqi_delay=?"
    axs[0].set_title(f"Downlink LA (slot axis)  |  {delay_txt}")
    axs[0].legend(loc="best", fontsize=8)
    axs[0].grid(True, alpha=0.3)
    _flush_xlim(axs[0], x_max)

    cqi_slot = _slot_hold(ref["cqi_index"], ref["num_slots"], num_slots)
    axs[1].step(slots, cqi_slot, where="post", label="reported CQI (TB hold)", color="C2")
    axs[1].set_ylabel("CQI index")
    axs[1].set_ylim(-0.5, 15.5)
    axs[1].legend(loc="best", fontsize=8)
    axs[1].grid(True, alpha=0.3)
    _flush_xlim(axs[1], x_max)

    for i, (name, res) in enumerate(results.items()):
        mcs_slot = _slot_hold(res["mcs_used"], res["num_slots"], num_slots)
        axs[2].step(
            slots,
            mcs_slot,
            where="post",
            color=f"C{i}",
            alpha=0.35,
            lw=0.8,
            label=f"{name.upper()} MCS",
        )
        axs[2].plot(
            slots,
            _rolling(mcs_slot, mcs_window),
            color=f"C{i}",
            lw=1.6,
            label=f"{name.upper()} {mcs_window}-slot mean",
        )
    axs[2].set_ylabel("MCS")
    axs[2].legend(loc="best", fontsize=7, ncol=2)
    axs[2].grid(True, alpha=0.3)
    _flush_xlim(axs[2], x_max)

    for name, res in results.items():
        y = _slot_cum_return(res["reward"], res["num_slots"], num_slots)
        axs[3].plot(slots, y, label=name.upper())
    axs[3].set_ylabel("cum reward")
    axs[3].legend(loc="best", fontsize=8)
    axs[3].grid(True, alpha=0.3)
    _flush_xlim(axs[3], x_max)

    for i, (name, res) in enumerate(results.items()):
        ack = res["ack"].astype(np.float64)
        emp = 1.0 - _running_mean(ack)
        emp_slot = _slot_hold(emp, res["num_slots"], num_slots)
        axs[4].plot(slots, emp_slot, color=f"C{i}", lw=1.5, label=f"{name.upper()} emp 1st-tx")
        pred = _running_mean(res["tbler"])
        pred_slot = _slot_hold(pred, res["num_slots"], num_slots)
        axs[4].plot(
            slots,
            pred_slot,
            color=f"C{i}",
            ls=":",
            lw=1.3,
            label=f"{name.upper()} pred TBLER",
        )
    axs[4].axhline(bler_target, color="k", ls="--", label="target")
    axs[4].set_ylabel("first-tx BLER")
    axs[4].set_xlabel("slot")
    axs[4].set_ylim(-0.02, 1.02)
    axs[4].legend(loc="best", fontsize=7, ncol=2)
    axs[4].grid(True, alpha=0.3)
    _flush_xlim(axs[4], x_max)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "downlink_la.yaml",
    )
    p.add_argument("--num-slots", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--eval-seeds", type=int, nargs="+", default=None)
    p.add_argument("--epsilons", type=float, nargs="+", default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--no-npz", action="store_true")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.num_slots is not None:
        cfg["num_slots"] = args.num_slots
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.epsilons is not None:
        cfg["collect_epsilons"] = args.epsilons
    if args.out_dir is not None:
        cfg["out_dir"] = str(args.out_dir)
    if args.no_plot:
        cfg["save_plot"] = False
    if args.no_npz:
        cfg["save_npz"] = False

    eval_seeds = (
        [int(s) for s in args.eval_seeds]
        if args.eval_seeds
        else [int(s) for s in cfg["eval_seeds"]]
        if cfg.get("eval_seeds")
        else [int(cfg["seed"])]
    )
    num_slots = int(cfg["num_slots"])
    bler_target = float(cfg["bler_target"])
    olla_step_up_db = cfg.get("olla_step_up_db")
    epsilons = [float(e) for e in cfg.get("collect_epsilons", [0.0])]

    print("=== Decision-step env + PHYAbstraction ===")
    print(f"num_slots={num_slots} eval_seeds={eval_seeds}")
    phy_abs = PHYAbstraction()
    env = DownlinkLAEnv.from_config(cfg, phy_abs=phy_abs)
    n_actions = env.action_space.n
    print(f"state={env.state_space.shape} actions={n_actions}")

    baselines = {
        "illa": make_baseline_policy("illa", env, bler_target=bler_target),
        "olla": make_baseline_policy(
            "olla", env, bler_target=bler_target, olla_step_up_db=olla_step_up_db
        ),
    }

    out_dir = Path(cfg["out_dir"])
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    illa_rows, olla_rows = [], []
    plot_seed = eval_seeds[0]

    for eval_seed in eval_seeds:
        np.random.seed(eval_seed)
        print(f"\n--- seed {eval_seed} ---")

        results = {}
        for eps in epsilons:
            for name, base in baselines.items():
                if eps == 0.0:
                    policy, label = base, name
                else:
                    policy = EpsilonGreedyPolicy(base, n_actions, epsilon=eps, seed=eval_seed)
                    label = f"{name}_eps{eps:g}"
                print(f"=== Rollout: {label} ===")
                seed_phy(eval_seed)
                res = rollout(env, policy, seed=eval_seed)
                m = summarize(label, res, bler_target, num_slots)
                results[label] = res
                if label == "illa":
                    illa_rows.append(m)
                elif label == "olla":
                    olla_rows.append(m)

        if cfg.get("save_plot", True) and eval_seed == plot_seed:
            path = out_dir / f"la_baselines_seed{eval_seed}.png"
            plot_results(results, bler_target, path, num_slots=num_slots)
            print(f"Saved plot -> {path}")

        if cfg.get("save_npz", True):
            for label, res in results.items():
                path = out_dir / f"la_{label}_seed{eval_seed}.npz"
                np.savez_compressed(
                    path,
                    **res,
                    bler_target=bler_target,
                    seed=eval_seed,
                    episode_num_slots=num_slots,
                )
                print(f"Saved {len(res['action'])} TB transitions -> {path}")

    if len(eval_seeds) > 1:
        print("\n=== Eval mean ± std ---")
        illa_agg = aggregate_metrics(illa_rows)
        olla_agg = aggregate_metrics(olla_rows)
        print_mean_row("ILLA", illa_agg, bler_target)
        print_mean_row("OLLA", olla_agg, bler_target)

        summary_path = out_dir / f"la_baselines_eval_{len(eval_seeds)}seeds.npz"
        np.savez_compressed(
            summary_path,
            eval_seeds=np.asarray(eval_seeds, dtype=np.int64),
            num_slots=num_slots,
            bler_target=bler_target,
            illa_return=np.asarray([r["return"] for r in illa_rows]),
            illa_throughput=np.asarray([r["throughput"] for r in illa_rows]),
            illa_first_tx_bler=np.asarray([r["first_tx_bler"] for r in illa_rows]),
            olla_return=np.asarray([r["return"] for r in olla_rows]),
            olla_throughput=np.asarray([r["throughput"] for r in olla_rows]),
            olla_first_tx_bler=np.asarray([r["first_tx_bler"] for r in olla_rows]),
            **{f"illa_{k}": v for k, v in illa_agg.items()},
            **{f"olla_{k}": v for k, v in olla_agg.items()},
        )
        print(f"Saved eval summary -> {summary_path}")

    print("Done.")


if __name__ == "__main__":
    main()
