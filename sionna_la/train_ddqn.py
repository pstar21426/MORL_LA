# Train / eval DDQN on DownlinkLAEnv; compare with ILLA / OLLA

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from sionna.sys import PHYAbstraction

from ddqn import DDQNAgent, Transition
from la_env import DownlinkLAEnv, seed_phy
from policies import make_baseline_policy


def load_config(path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _metrics(st, num_slots):
    n = max(st["tbs"], 1)
    n_fin = max(st["tbs"] - st["n_trunc"], 1)
    return {
        "return": st["ret"],
        "tbs": st["tbs"],
        "first_tx_bler": 1.0 - st["n_ack"] / n,
        "tb_fail": 1.0 - st["n_ok"] / n_fin,
        "mean_mcs": st["mcs_sum"] / n,
        "mean_slots_tb": st["n_slots"] / n,
        "throughput": st["ret"] / max(num_slots, 1),
        "drops": st["drops"],
    }


def _rollout(env, choose_action, seed, on_transition=None):
    state, info = env.reset(seed=seed)
    st = dict(ret=0.0, tbs=0, n_ack=0, n_slots=0, n_ok=0, n_trunc=0, mcs_sum=0.0, drops=0)
    done = False
    while not done:
        action = choose_action(state, info)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if on_transition is not None:
            on_transition(state, action, reward, next_state, terminated, truncated)
        out = info["outcome"]
        st["ret"] += reward
        st["tbs"] += 1
        st["n_ack"] += int(out["ack"])
        st["n_slots"] += int(out["num_slots"])
        st["n_ok"] += int(out["tb_success"])
        st["n_trunc"] += int(out["truncated_mid_tb"])
        st["mcs_sum"] += float(out["mcs_used"])
        st["drops"] += int(out["dropped"])
        state = next_state
    return _metrics(st, env.num_slots)


def run_episode(env, agent, seed, *, train=True, greedy=False):
    def on_transition(state, action, reward, next_state, terminated, truncated):
        if not train:
            return
        # time-limit truncated is not a true terminal; still bootstrap Q
        del truncated
        agent.push(
            Transition(
                state=state,
                action=action,
                reward=float(reward),
                next_state=next_state,
                terminated=bool(terminated),
            )
        )
        agent.train_step()

    return _rollout(
        env,
        lambda state, _info: agent.select_action(state, greedy=greedy),
        seed,
        on_transition=on_transition,
    )


def rollout_rule(env, policy, seed):
    policy.reset()
    return _rollout(env, policy, seed)


def print_row(name, m, bler_target, *, prefix=""):
    print(
        f"{prefix}[{name:>5}] return={m['return']:7.1f} | "
        f"throughput={m['throughput']:.4f} | "
        f"TBs={m['tbs']:3d} | "
        f"1st-tx BLER={m['first_tx_bler']:.3f} (tgt {bler_target}) | "
        f"slots/TB={m['mean_slots_tb']:.2f} | drops={m['drops']}"
    )


def print_mean_row(name, metrics, bler_target):
    print(
        f"[{name:>5}] return={metrics['return_mean']:7.1f} ± {metrics['return_std']:.1f} | "
        f"throughput={metrics['throughput_mean']:.4f} ± {metrics['throughput_std']:.4f} | "
        f"TBs={metrics['tbs_mean']:.0f} | "
        f"1st-tx BLER={metrics['first_tx_bler_mean']:.3f} (tgt {bler_target}) | "
        f"slots/TB={metrics['mean_slots_tb_mean']:.2f} | drops={metrics['drops_mean']:.1f}"
    )


def aggregate_metrics(rows):
    keys = ["return", "throughput", "tbs", "first_tx_bler", "mean_slots_tb", "drops"]
    out = {}
    for k in keys:
        vals = np.asarray([r[k] for r in rows], dtype=np.float64)
        out[f"{k}_mean"] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std())
    return out


def held_out_eval_seeds(train_seed, episodes, requested):
    """Training rollouts use env seeds [train_seed, train_seed + episodes)."""
    used = set(range(int(train_seed), int(train_seed) + int(episodes)))
    step = max(int(episodes), 1)
    out = []
    for s in requested:
        s = int(s)
        while s in used:
            s += step
        used.add(s)
        out.append(s)
    return out


def resolve_held_out_eval_seed(cfg, requested):
    train_seed = int(cfg.get("seed", 0))
    episodes = int((cfg.get("train") or {}).get("episodes", 1000))
    requested = int(requested)
    resolved = held_out_eval_seeds(train_seed, episodes, [requested])[0]
    return resolved, train_seed, episodes


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "downlink_la.yaml",
    )
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--eval-seeds", type=int, nargs="+", default=None)
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--buffer-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epsilon-start", type=float, default=None)
    p.add_argument("--epsilon-end", type=float, default=None)
    p.add_argument("--log-every", type=int, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg.get("train") or {}
    ddqn_cfg = cfg.get("ddqn") or {}

    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))
    episodes = int(
        args.episodes if args.episodes is not None else train_cfg.get("episodes", 1000)
    )
    log_every = int(
        args.log_every if args.log_every is not None else train_cfg.get("log_every", 50)
    )
    hidden = int(args.hidden if args.hidden is not None else ddqn_cfg.get("hidden", 256))
    buffer_size = int(
        args.buffer_size
        if args.buffer_size is not None
        else ddqn_cfg.get("buffer_size", 100_000)
    )
    batch_size = int(
        args.batch_size if args.batch_size is not None else ddqn_cfg.get("batch_size", 64)
    )
    gamma = float(args.gamma if args.gamma is not None else ddqn_cfg.get("gamma", 0.99))
    lr = float(args.lr if args.lr is not None else ddqn_cfg.get("lr", 1e-3))
    epsilon_start = float(
        args.epsilon_start
        if args.epsilon_start is not None
        else ddqn_cfg.get("epsilon_start", 1.0)
    )
    epsilon_end = float(
        args.epsilon_end
        if args.epsilon_end is not None
        else ddqn_cfg.get("epsilon_end", 0.01)
    )
    decay_cfg = ddqn_cfg.get("epsilon_decay_steps")
    decay_steps = int(decay_cfg) if decay_cfg is not None else max(episodes * 100, 5_000)

    np.random.seed(seed)
    seed_phy(seed)
    bler_target = float(cfg["bler_target"])
    olla_step_up_db = cfg.get("olla_step_up_db")

    out_dir = args.out_dir or Path(cfg.get("out_dir", "outputs"))
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    phy_abs = PHYAbstraction()
    env = DownlinkLAEnv.from_config(cfg, phy_abs=phy_abs)
    state_dim = int(np.prod(env.state_space.shape))
    n_actions = env.action_space.n

    agent = DDQNAgent(
        state_dim,
        n_actions,
        hidden=hidden,
        gamma=gamma,
        lr=lr,
        buffer_size=buffer_size,
        batch_size=batch_size,
        target_sync=int(ddqn_cfg.get("target_sync", 200)),
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_steps=decay_steps,
    )

    print(
        f"=== DDQN train ({episodes} eps) state={state_dim} "
        f"actions={n_actions} hidden={hidden} "
        f"buffer={buffer_size} eps_end={epsilon_end} ==="
    )
    train_log = {
        "episode": [],
        "return": [],
        "throughput": [],
        "tbs": [],
        "first_tx_bler": [],
        "epsilon": [],
        "mean_mcs": [],
        "drops": [],
    }

    for ep in range(episodes):
        m = run_episode(env, agent, seed=seed + ep, train=True)
        train_log["episode"].append(ep + 1)
        train_log["return"].append(m["return"])
        train_log["throughput"].append(m["throughput"])
        train_log["tbs"].append(m["tbs"])
        train_log["first_tx_bler"].append(m["first_tx_bler"])
        train_log["epsilon"].append(agent.epsilon)
        train_log["mean_mcs"].append(m["mean_mcs"])
        train_log["drops"].append(m["drops"])

        if (ep + 1) % log_every == 0 or ep == 0:
            print(
                f"  ep {ep + 1:4d} | return={m['return']:.1f} | "
                f"throughput={m['throughput']:.4f} | eps={agent.epsilon:.3f} | "
                f"TBs={m['tbs']} | 1st-tx BLER={m['first_tx_bler']:.3f} | "
                f"MCS={m['mean_mcs']:.1f} | drops={m['drops']}"
            )

    log_path = out_dir / f"ddqn_train_log_seed{seed}.npz"
    np.savez_compressed(log_path, **{k: np.asarray(v) for k, v in train_log.items()})
    print(f"Saved training log -> {log_path}")

    ckpt_path = out_dir / f"ddqn_seed{seed}.pt"
    torch.save(
        {
            "q": agent.q.state_dict(),
            "state_dim": state_dim,
            "n_actions": n_actions,
            "hidden": hidden,
            "episodes": episodes,
            "epsilon_end": epsilon_end,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint -> {ckpt_path}")

    if args.eval_seeds is not None:
        requested_eval = [int(s) for s in args.eval_seeds]
    else:
        requested_eval = [int(s) for s in train_cfg.get("eval_seeds", list(range(10)))]
    eval_seeds = held_out_eval_seeds(seed, episodes, requested_eval)
    if eval_seeds != requested_eval:
        print(
            f"eval seeds {requested_eval} overlap training "
            f"[{seed}, {seed + episodes}); using {eval_seeds}"
        )
    print(f"\n=== Eval ({len(eval_seeds)} seeds: {eval_seeds}) ===")

    illa_pol = make_baseline_policy("illa", env, bler_target=bler_target)
    olla_pol = make_baseline_policy(
        "olla", env, bler_target=bler_target, olla_step_up_db=olla_step_up_db
    )

    ddqn_rows, illa_rows, olla_rows = [], [], []
    for eval_seed in eval_seeds:
        seed_phy(eval_seed)
        ddqn_m = run_episode(env, agent, seed=eval_seed, train=False, greedy=True)
        seed_phy(eval_seed)
        illa_m = rollout_rule(env, illa_pol, eval_seed)
        seed_phy(eval_seed)
        olla_m = rollout_rule(env, olla_pol, eval_seed)
        ddqn_rows.append(ddqn_m)
        illa_rows.append(illa_m)
        olla_rows.append(olla_m)
        print(f"--- seed {eval_seed} ---")
        print_row("DDQN", ddqn_m, bler_target)
        print_row("ILLA", illa_m, bler_target)
        print_row("OLLA", olla_m, bler_target)

    print("\n=== Eval mean ± std ---")
    ddqn_agg = aggregate_metrics(ddqn_rows)
    illa_agg = aggregate_metrics(illa_rows)
    olla_agg = aggregate_metrics(olla_rows)
    print_mean_row("DDQN", ddqn_agg, bler_target)
    print_mean_row("ILLA", illa_agg, bler_target)
    print_mean_row("OLLA", olla_agg, bler_target)

    eval_path = out_dir / f"ddqn_eval_seeds{seed}.npz"
    np.savez_compressed(
        eval_path,
        eval_seeds=np.asarray(eval_seeds, dtype=np.int64),
        ddqn_return=np.asarray([r["return"] for r in ddqn_rows]),
        ddqn_throughput=np.asarray([r["throughput"] for r in ddqn_rows]),
        illa_return=np.asarray([r["return"] for r in illa_rows]),
        illa_throughput=np.asarray([r["throughput"] for r in illa_rows]),
        olla_return=np.asarray([r["return"] for r in olla_rows]),
        olla_throughput=np.asarray([r["throughput"] for r in olla_rows]),
        bler_target=bler_target,
        seed=seed,
        episodes=episodes,
        hidden=hidden,
        epsilon_end=epsilon_end,
    )
    print(f"Saved eval summary -> {eval_path}")

    md_path = out_dir / f"ddqn_eval_table_seed{seed}.md"
    lines = [
        f"# DDQN eval vs ILLA / OLLA (train seed={seed}, {episodes} ep)",
        "",
        "| seed | DDQN ret | ILLA ret | OLLA ret | DDQN BLER | ILLA BLER | OLLA BLER | DDQN MCS | ILLA MCS | OLLA MCS | DDQN drops | ILLA drops | OLLA drops |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for i, s in enumerate(eval_seeds):
        d, a, o = ddqn_rows[i], illa_rows[i], olla_rows[i]
        lines.append(
            f"| {s} | {d['return']:.1f} | {a['return']:.1f} | {o['return']:.1f} | "
            f"{d['first_tx_bler']:.3f} | {a['first_tx_bler']:.3f} | {o['first_tx_bler']:.3f} | "
            f"{d['mean_mcs']:.1f} | {a['mean_mcs']:.1f} | {o['mean_mcs']:.1f} | "
            f"{d['drops']} | {a['drops']} | {o['drops']} |"
        )
    lines += [
        "",
        f"**mean±std return** DDQN {ddqn_agg['return_mean']:.1f}±{ddqn_agg['return_std']:.1f} · "
        f"ILLA {illa_agg['return_mean']:.1f}±{illa_agg['return_std']:.1f} · "
        f"OLLA {olla_agg['return_mean']:.1f}±{olla_agg['return_std']:.1f}",
        "",
        f"**mean 1st-tx BLER** DDQN {ddqn_agg['first_tx_bler_mean']:.3f} · "
        f"ILLA {illa_agg['first_tx_bler_mean']:.3f} · "
        f"OLLA {olla_agg['first_tx_bler_mean']:.3f}",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved eval table -> {md_path}")
    print("Done.")


if __name__ == "__main__":
    main()
