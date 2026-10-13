# Train the paper ensemble DQN and run the same uncertainty diagnostics.

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from sionna.sys import PHYAbstraction

from la_env import DownlinkLAEnv, seed_phy
from policies import make_baseline_policy
from train_uq import held_out_eval_seeds, rollout_metrics, summarize_uq
from uq_ensemble import EnsembleDQNAgent


def load_config(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "downlink_la.yaml",
    )
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--n-models", type=int, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg.get("train") or {}
    uq_cfg = cfg.get("uq_dqn") or {}
    ens_cfg = cfg.get("ensemble") or {}

    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))
    episodes = int(args.episodes if args.episodes is not None else train_cfg.get("episodes", 80))
    log_every = int(train_cfg.get("log_every", 20))
    n_models = int(
        args.n_models if args.n_models is not None else ens_cfg.get("n_models", 5)
    )

    out_dir = args.out_dir or Path(cfg.get("out_dir", "outputs"))
    if not out_dir.is_absolute():
        out_dir = Path(__file__).resolve().parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    seed_phy(seed)
    phy_abs = PHYAbstraction()
    env = DownlinkLAEnv.from_config(cfg, phy_abs=phy_abs)
    state_dim = int(np.prod(env.state_space.shape))
    n_actions = env.action_space.n
    rho_ema = float(uq_cfg.get("rho_ema", 0.1))

    agent = EnsembleDQNAgent(
        state_dim,
        n_actions,
        n_models=n_models,
        hidden=int(ens_cfg.get("hidden", uq_cfg.get("hidden", 256))),
        gamma=float(uq_cfg.get("gamma", 0.7)),
        lr=float(uq_cfg.get("lr", 1e-3)),
        buffer_size=int(uq_cfg.get("buffer_size", 100_000)),
        batch_size=int(uq_cfg.get("batch_size", 64)),
        target_sync=int(uq_cfg.get("target_sync", 200)),
        epsilon_start=float(uq_cfg.get("epsilon_start", 1.0)),
        epsilon_end=float(uq_cfg.get("epsilon_end", 0.01)),
        epsilon_decay_steps=int(uq_cfg.get("epsilon_decay_steps", 30_000)),
        rho=0.0,
        seed=seed,
    )
    print(
        f"=== Ensemble DQN K={n_models} train ({episodes} ep) "
        f"gamma={agent.gamma} device={agent.device} queue_feat={env.queue_feat} ==="
    )
    print("u_alea = mean sigma_k, u_epist = mean (mu_k - mu)^2, loss = Gaussian NLL")

    for ep in range(episodes):
        m, _ = rollout_metrics(env, agent, seed=seed + ep, train=True)
        ep_rate = m["return_per_slot"]
        agent.rho = ep_rate if ep == 0 else (1.0 - rho_ema) * agent.rho + rho_ema * ep_rate
        if (ep + 1) % log_every == 0 or ep == 0:
            print(
                f"  ep {ep + 1:4d} | return={m['ret']:.1f} | "
                f"rate={ep_rate:.4f} | rho={agent.rho:.4f} | "
                f"eps={agent.epsilon:.3f} | nll={agent.last_nll:.4f} | "
                f"u_alea={agent.last_u_alea:.4f} | u_epist={agent.last_u_epist:.4f}"
            )

    ckpt = out_dir / f"uq_ensemble_k{n_models}_seed{seed}.pt"
    torch.save(
        {
            **agent.state_dict(),
            "state_dim": state_dim,
            "n_actions": n_actions,
            "hidden": int(ens_cfg.get("hidden", uq_cfg.get("hidden", 256))),
            "gamma": agent.gamma,
            "rho": agent.rho,
            "episodes": episodes,
        },
        ckpt,
    )
    print(f"Saved {ckpt}")

    eval_seeds = held_out_eval_seeds(
        seed, episodes, [int(s) for s in train_cfg.get("eval_seeds", [205, 206, 207])]
    )
    olla = make_baseline_policy(
        "olla",
        env,
        bler_target=float(cfg["bler_target"]),
        olla_step_up_db=cfg.get("olla_step_up_db"),
    )

    id_rows, ood_rows, returns, olla_returns = [], [], [], []
    cfg_ood = dict(cfg)
    cfg_ood["cqi_bias_db"] = 0.0
    env_id = DownlinkLAEnv.from_config(cfg, phy_abs=phy_abs)
    env_ood = DownlinkLAEnv.from_config(cfg_ood, phy_abs=phy_abs)

    for es in eval_seeds:
        seed_phy(es)
        m_id, uq_id = rollout_metrics(
            env_id, agent, seed=es, train=False, greedy=True, collect_uq=True
        )
        seed_phy(es)
        m_ood, uq_ood = rollout_metrics(
            env_ood, agent, seed=es, train=False, greedy=True, collect_uq=True
        )
        seed_phy(es)
        olla.reset()
        state, info = env_id.reset(seed=es)
        olla_ret = 0.0
        done = False
        while not done:
            action = olla(state, info)
            state, reward, terminated, truncated, info = env_id.step(action)
            olla_ret += float(reward)
            done = terminated or truncated
        returns.append(m_id["ret"])
        olla_returns.append(olla_ret)
        id_rows.extend(uq_id)
        ood_rows.extend(uq_ood)
        print(
            f"seed {es}: ensemble={m_id['ret']:.1f} OOD(bias=0)={m_ood['ret']:.1f} "
            f"OLLA={olla_ret:.1f}"
        )

    sum_id = summarize_uq(id_rows)
    u_ep_id = np.asarray([r["u_epist"] for r in id_rows], dtype=np.float64)
    u_ep_ood = np.asarray([r["u_epist"] for r in ood_rows], dtype=np.float64)
    report = [
        f"# Ensemble DQN (K={n_models}, train seed={seed}, {episodes} ep)",
        "",
        "Paper aggregation: u_alea = mean_k sigma_k, u_epist = mean_k (mu_k - mu)^2.",
        "Each member is a Gaussian Q-network trained with NLL on the DQN target.",
        "The executed policy is argmax of the mean mu. Reward is r - rho * tau, gamma=0.7.",
        "",
        f"- Mean eval return (ID): **{np.mean(returns):.1f}** vs OLLA **{np.mean(olla_returns):.1f}** "
        f"(delta {np.mean(returns) - np.mean(olla_returns):+.1f})",
        f"- Epistemic mean ID vs OOD (cqi_bias 3→0): **{u_ep_id.mean():.4f}** vs **{u_ep_ood.mean():.4f}**",
        f"- Aleatoric mean if num_retx>0 vs ==0: **{sum_id.get('alea_retx_gt0', float('nan')):.4f}** vs "
        f"**{sum_id.get('alea_retx_eq0', float('nan')):.4f}** "
        f"(Spearman vs retx {sum_id.get('rho_alea_retx', float('nan')):.3f})",
        f"- Epistemic vs 1/sqrt(visit): Spearman **{sum_id.get('rho_epist_inv_visit', float('nan')):.3f}**",
    ]
    md_path = out_dir / f"uq_ensemble_k{n_models}_report_seed{seed}.md"
    md_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    np.savez_compressed(
        out_dir / f"uq_ensemble_k{n_models}_diag_seed{seed}.npz",
        id_u_alea=np.asarray([r["u_alea"] for r in id_rows]),
        id_u_epist=np.asarray([r["u_epist"] for r in id_rows]),
        id_retx=np.asarray([r.get("num_retx", 0) for r in id_rows]),
        ood_u_epist=u_ep_ood,
        eval_returns=np.asarray(returns),
        olla_returns=np.asarray(olla_returns),
        **{f"sum_id_{k}": v for k, v in sum_id.items()},
    )
    print(f"Wrote {md_path}")
    print("\n".join(report[6:]))


if __name__ == "__main__":
    main()
