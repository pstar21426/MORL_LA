# Train / eval a simple DQN on DownlinkLAEnv; compare with ILLA / OLLA

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from sionna.phy import config as sionna_config
from sionna.sys import PHYAbstraction

from dqn import DQNAgent, Transition
from la_env import DownlinkLAEnv
from policies import make_baseline_policy


def load_config(path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_episode(env, agent, seed, *, train=True, greedy=False):
    state, info = env.reset(seed=seed)
    ep_return = 0.0
    n_tbs = 0
    n_ack = 0
    n_slots = 0
    n_tb_success = 0
    mcs_sum = 0.0
    drops = 0

    done = False
    while not done:
        action = agent.select_action(state, greedy=greedy)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if train:
            agent.push(
                Transition(
                    state=state,
                    action=action,
                    reward=float(reward),
                    next_state=next_state,
                    done=done,
                )
            )
            agent.train_step()

        ep_return += reward
        n_tbs += 1
        n_ack += int(info["ack"])
        n_slots += int(info["num_slots"])
        n_tb_success += int(info["tb_success"])
        mcs_sum += float(info["mcs_used"])
        drops += int(info["dropped"])
        state = next_state

    return {
        "return": ep_return,
        "tbs": n_tbs,
        "first_tx_bler": 1.0 - n_ack / max(n_tbs, 1),
        "tb_fail": 1.0 - n_tb_success / max(n_tbs, 1),
        "mean_mcs": mcs_sum / max(n_tbs, 1),
        "mean_slots_tb": n_slots / max(n_tbs, 1),
        "throughput": ep_return / max(env.num_slots, 1),
        "drops": drops,
    }


def rollout_rule(env, policy, seed):
    state, info = env.reset(seed=seed)
    policy.reset()
    ep_return = 0.0
    n_tbs = 0
    n_ack = 0
    n_slots = 0
    n_tb_success = 0
    mcs_sum = 0.0
    drops = 0

    done = False
    while not done:
        action = policy(state, info)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_return += reward
        n_tbs += 1
        n_ack += int(info["ack"])
        n_slots += int(info["num_slots"])
        n_tb_success += int(info["tb_success"])
        mcs_sum += float(info["mcs_used"])
        drops += int(info["dropped"])

    return {
        "return": ep_return,
        "tbs": n_tbs,
        "first_tx_bler": 1.0 - n_ack / max(n_tbs, 1),
        "tb_fail": 1.0 - n_tb_success / max(n_tbs, 1),
        "mean_mcs": mcs_sum / max(n_tbs, 1),
        "mean_slots_tb": n_slots / max(n_tbs, 1),
        "throughput": ep_return / max(env.num_slots, 1),
        "drops": drops,
    }


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "downlink_la.yaml",
    )
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--buffer-size", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epsilon-end", type=float, default=0.01)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    seed = args.seed
    sionna_config.seed = seed
    torch.manual_seed(seed)
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

    decay_steps = max(args.episodes * 100, 5_000)
    agent = DQNAgent(
        state_dim,
        n_actions,
        hidden=args.hidden,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=decay_steps,
    )

    print(
        f"=== DQN train ({args.episodes} eps) state={state_dim} "
        f"actions={n_actions} hidden={args.hidden} "
        f"buffer={args.buffer_size} eps_end={args.epsilon_end} ==="
    )
    train_log = {
        "episode": [],
        "return": [],
        "throughput": [],
        "tbs": [],
        "first_tx_bler": [],
        "epsilon": [],
    }

    for ep in range(args.episodes):
        m = run_episode(env, agent, seed=seed + ep, train=True)
        train_log["episode"].append(ep + 1)
        train_log["return"].append(m["return"])
        train_log["throughput"].append(m["throughput"])
        train_log["tbs"].append(m["tbs"])
        train_log["first_tx_bler"].append(m["first_tx_bler"])
        train_log["epsilon"].append(agent.epsilon)

        if (ep + 1) % args.log_every == 0 or ep == 0:
            print(
                f"  ep {ep + 1:4d} | return={m['return']:.1f} | "
                f"throughput={m['throughput']:.4f} | eps={agent.epsilon:.3f} | "
                f"TBs={m['tbs']} | 1st-tx BLER={m['first_tx_bler']:.3f}"
            )

    log_path = out_dir / f"dqn_train_log_seed{seed}.npz"
    np.savez_compressed(log_path, **{k: np.asarray(v) for k, v in train_log.items()})
    print(f"Saved training log -> {log_path}")

    ckpt_path = out_dir / f"dqn_seed{seed}.pt"
    torch.save(
        {
            "q": agent.q.state_dict(),
            "state_dim": state_dim,
            "n_actions": n_actions,
            "hidden": args.hidden,
            "episodes": args.episodes,
            "epsilon_end": args.epsilon_end,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint -> {ckpt_path}")

    eval_seeds = [int(s) for s in args.eval_seeds]
    print(f"\n=== Eval ({len(eval_seeds)} seeds: {eval_seeds}) ===")

    dqn_rows, illa_rows, olla_rows = [], [], []
    for eval_seed in eval_seeds:
        sionna_config.seed = eval_seed
        torch.manual_seed(eval_seed)

        dqn_m = run_episode(env, agent, seed=eval_seed, train=False, greedy=True)
        illa_m = rollout_rule(
            env, make_baseline_policy("illa", env, bler_target=bler_target), eval_seed
        )
        olla_m = rollout_rule(
            env,
            make_baseline_policy(
                "olla", env, bler_target=bler_target, olla_step_up_db=olla_step_up_db
            ),
            eval_seed,
        )
        dqn_rows.append(dqn_m)
        illa_rows.append(illa_m)
        olla_rows.append(olla_m)
        print(f"--- seed {eval_seed} ---")
        print_row("DQN", dqn_m, bler_target)
        print_row("ILLA", illa_m, bler_target)
        print_row("OLLA", olla_m, bler_target)

    print("\n=== Eval mean ± std ---")
    dqn_agg = aggregate_metrics(dqn_rows)
    illa_agg = aggregate_metrics(illa_rows)
    olla_agg = aggregate_metrics(olla_rows)
    print_mean_row("DQN", dqn_agg, bler_target)
    print_mean_row("ILLA", illa_agg, bler_target)
    print_mean_row("OLLA", olla_agg, bler_target)

    eval_path = out_dir / f"dqn_eval_seeds{seed}.npz"
    np.savez_compressed(
        eval_path,
        eval_seeds=np.asarray(eval_seeds, dtype=np.int64),
        dqn_return=np.asarray([r["return"] for r in dqn_rows]),
        dqn_throughput=np.asarray([r["throughput"] for r in dqn_rows]),
        illa_return=np.asarray([r["return"] for r in illa_rows]),
        illa_throughput=np.asarray([r["throughput"] for r in illa_rows]),
        olla_return=np.asarray([r["return"] for r in olla_rows]),
        olla_throughput=np.asarray([r["throughput"] for r in olla_rows]),
        bler_target=bler_target,
        seed=seed,
        episodes=args.episodes,
        hidden=args.hidden,
        epsilon_end=args.epsilon_end,
    )
    print(f"Saved eval summary -> {eval_path}")
    print("Done.")


if __name__ == "__main__":
    main()
