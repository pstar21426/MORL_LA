import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from sionna.sys import PHYAbstraction

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from la_env import DownlinkLAEnv, seed_phy
from policies import make_baseline_policy

_UNSEEN = -1.0


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _state_slices(L: int):
    # [cqi_n, past_cqi x L, past_m x L, past_b x L]
    i_cqi = 1 + L
    i_m = i_cqi + L
    i_b = i_m + L
    return slice(0, 1), slice(1, i_cqi), slice(i_cqi, i_m), slice(i_m, i_b)


def rollout(env: DownlinkLAEnv, policy, seed: int) -> dict:
    state, info = env.reset(seed=seed)
    policy.reset()
    L = env.hist.num_lags

    log = {
        "state": [],
        "next_state": [],
        "decision_slot": [],
        "cqi_index": [],
        "mcs_norm": [],
        "ack": [],
        "sinr_true_decision": [],
        "sinr_hat_decision": [],
    }
    done = False
    while not done:
        log["state"].append(state.copy())
        log["decision_slot"].append(int(info["slot"]))
        log["cqi_index"].append(int(info["cqi_index"]))
        log["sinr_true_decision"].append(float(info["sinr_true_db"]))
        log["sinr_hat_decision"].append(float(info["sinr_hat_db"]))

        action = policy(state, info)
        next_state, _r, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        out = info["outcome"]
        mcs_used = int(out["mcs_used"])
        mcs_norm = (mcs_used - env.mcs_min) / env._mcs_span

        log["next_state"].append(next_state.copy())
        log["mcs_norm"].append(mcs_norm)
        log["ack"].append(int(out["ack"]))
        state = next_state

    return {
        "state": np.asarray(log["state"], dtype=np.float32),
        "next_state": np.asarray(log["next_state"], dtype=np.float32),
        "decision_slot": np.asarray(log["decision_slot"], dtype=int),
        "cqi_index": np.asarray(log["cqi_index"], dtype=int),
        "mcs_norm": np.asarray(log["mcs_norm"], dtype=float),
        "ack": np.asarray(log["ack"], dtype=int),
        "sinr_true_decision": np.asarray(log["sinr_true_decision"], dtype=float),
        "sinr_hat_decision": np.asarray(log["sinr_hat_decision"], dtype=float),
        "sinr_true_db": env._sinr_true_db.copy(),
        "sinr_hat_db": env._sinr_fb_db.copy(),
        "cqi_delay_slots": int(env._delay_used),
        "cqi_noise_std_db": float(env.cqi_noise_std_db),
        "num_slots": int(env.num_slots),
        "ack_delay": int(env.ack_delay_slots),
        "state_num_lags": L,
        "seed": seed,
    }


def _cqi_on_slot_axis(decision_slot, cqi_index, num_slots):
    out = np.full(num_slots, np.nan, dtype=float)
    if len(decision_slot) == 0:
        return out
    for i, t0 in enumerate(decision_slot):
        t1 = decision_slot[i + 1] if i + 1 < len(decision_slot) else num_slots
        t0 = int(np.clip(t0, 0, num_slots))
        t1 = int(np.clip(t1, t0, num_slots))
        out[t0:t1] = float(cqi_index[i])
    return out


def check_csi_numbers(log: dict) -> dict:
    gamma = log["sinr_true_db"]
    hat = log["sinr_hat_db"]
    d = int(log["cqi_delay_slots"])
    noise = float(log["cqi_noise_std_db"])
    if d <= 0:
        resid = hat - gamma
    else:
        resid = hat[d:] - gamma[:-d]
    n = len(resid)
    return {
        "cqi_delay_slots": d,
        "noise_cfg_db": noise,
        "resid_mean_db": float(np.mean(resid)) if n else float("nan"),
        "resid_std_db": float(np.std(resid)) if n else float("nan"),
        "corr_aligned": (
            float(np.corrcoef(hat[d:] if d else hat, gamma[:-d] if d else gamma)[0, 1])
            if n > 1
            else float("nan")
        ),
        "n_aligned": n,
    }


def plot_csi_overlay(log: dict, policy_name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = int(log["num_slots"])
    t = np.arange(n)
    gamma = log["sinr_true_db"][:n]
    hat = log["sinr_hat_db"][:n]
    cqi_slot = _cqi_on_slot_axis(log["decision_slot"], log["cqi_index"], n)
    d = int(log["cqi_delay_slots"])
    noise = float(log["cqi_noise_std_db"])
    tag = f"{policy_name}_seed{log['seed']}"
    stats = check_csi_numbers(log)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(t, gamma, color="C0", lw=1.0, label=r"$\gamma_\mathrm{true}$")
    ax.plot(t, hat, color="C1", lw=1.0, alpha=0.9, label=r"$\hat{\gamma}$ (pre-CQI)")
    ax.set_xlabel("slot")
    ax.set_ylabel("SINR [dB]")
    ax.set_xlim(0, n - 1)
    ax.margins(x=0)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.step(t, cqi_slot, where="post", color="C2", lw=1.2, label="CQI index")
    ax2.set_ylabel("CQI index")
    ax2.set_ylim(-0.5, 15.5)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    ax.set_title(
        f"CSI overlay — {tag}  |  "
        f"cqi_delay={d}, noise={noise:g} dB, "
        f"corr={stats['corr_aligned']:.3f}, resid_std={stats['resid_std_db']:.2f} dB"
    )
    fig.tight_layout()
    path = out_dir / f"obs_csi_overlay_{tag}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _fmt(x, digits=4):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isclose(x, _UNSEEN))):
        return "-1"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{float(x):.{digits}f}"


def _print_table(title: str, headers: list[str], rows: list[list], max_rows: int):
    print(f"\n=== {title} ===")
    show = rows[:max_rows]
    widths = [len(h) for h in headers]
    for row in show:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("-+-".join("-" * w for w in widths))
    for row in show:
        print(" | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))
    if len(rows) > max_rows:
        print(f"... ({len(rows) - max_rows} more rows)")


def _write_csv(path: Path, headers: list[str], rows: list[list]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)


def _write_md_table(path: Path, title: str, headers: list[str], rows: list[list], max_rows: int | None = None):
    """Markdown table — opens as a real table in Cursor/GitHub preview."""
    path.parent.mkdir(parents=True, exist_ok=True)
    show = rows if max_rows is None else rows[:max_rows]
    lines = [
        f"# {title}",
        "",
        f"rows shown: {len(show)} / {len(rows)}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in show:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    if max_rows is not None and len(rows) > max_rows:
        lines.append("")
        lines.append(f"... ({len(rows) - max_rows} more rows; see CSV for full)")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_c_state0(log: dict, tol: float = 1e-5):
    # C: state[0](현재 CQI)가 cqi_index / 15와 일치하는지 확인
    headers = ["tb", "slot", "cqi_index", "state[0]", "cqi/15", "ok"]
    rows = []
    n_fail = 0
    for i in range(len(log["state"])):
        cqi = int(log["cqi_index"][i])
        s0 = float(log["state"][i, 0])
        expect = cqi / 15.0
        ok = abs(s0 - expect) <= tol
        if not ok:
            n_fail += 1
        rows.append(
            [i, int(log["decision_slot"][i]), cqi, _fmt(s0), _fmt(expect), "Y" if ok else "N"]
        )
    return headers, rows, n_fail


def table_d_history(log: dict, tol: float = 1e-5):
    # D: history(cqi,mcs,ack)들이 decision index와 확인하는지, 업데이트가 잘 되는지 확인
    L = int(log["state_num_lags"])
    _, sl_cqi, sl_m, sl_b = _state_slices(L)
    headers = [
        "tb",
        "slot",
        "lag",
        "past_cqi",
        "past_mcs",
        "past_ack",
        "exp_cqi",
        "exp_mcs",
        "exp_ack",
        "ok",
    ]
    rows = []
    n_fail = 0

    # 각 next_state(TB i) 에 대해, lag0 = TB i 의 outcome과 일치하는지 확인
    # 전체 lag table: decision i 에서, lag k 는 i-1-k >= 0 인 경우 TB (i-1-k)와 같아야 함
    n = len(log["state"])
    for i in range(n):
        st = log["state"][i]
        past_cqi = st[sl_cqi]
        past_mcs = st[sl_m]
        past_ack = st[sl_b]
        for lag in range(L):
            src = i - 1 - lag
            if src >= 0:
                exp_cqi = float(log["cqi_index"][src] / 15.0)
                exp_mcs = float(log["mcs_norm"][src])
                exp_ack = float(log["ack"][src])
            else:
                exp_cqi = exp_mcs = exp_ack = _UNSEEN

            got_cqi = float(past_cqi[lag])
            got_mcs = float(past_mcs[lag])
            got_ack = float(past_ack[lag])
            ok = (
                abs(got_cqi - exp_cqi) <= tol
                and abs(got_mcs - exp_mcs) <= tol
                and abs(got_ack - exp_ack) <= tol
            )
            if not ok:
                n_fail += 1
            rows.append(
                [
                    i,
                    int(log["decision_slot"][i]),
                    lag,
                    _fmt(got_cqi),
                    _fmt(got_mcs),
                    _fmt(got_ack),
                    _fmt(exp_cqi),
                    _fmt(exp_mcs),
                    _fmt(exp_ack),
                    "Y" if ok else "N",
                ]
            )
    return headers, rows, n_fail


def run(
    cfg: dict,
    seed: int,
    policy_name: str,
    num_slots: int | None,
    out_dir: Path,
    table_rows: int,
) -> bool:
    # C/D assume immediate history push
    cfg = {**cfg, "ack_delay_slots": 0}
    if num_slots is not None:
        cfg["num_slots"] = num_slots

    seed_phy(seed)
    phy_abs = PHYAbstraction()
    env = DownlinkLAEnv.from_config(cfg, phy_abs=phy_abs)
    if env.ack_delay_slots != 0:
        print("ERROR: ack_delay must be 0 for C/D tables")
        return False

    policy = make_baseline_policy(
        policy_name,
        env,
        bler_target=float(cfg["bler_target"]),
        olla_step_up_db=cfg.get("olla_step_up_db"),
    )
    log = rollout(env, policy, seed=seed)
    tag = f"{policy_name}_seed{seed}"

    # --- A: CSI 오버레이 ---
    stats = check_csi_numbers(log)
    plot_path = plot_csi_overlay(log, policy_name, out_dir)
    noise = stats["noise_cfg_db"]
    ok_a = True
    if stats["n_aligned"] > 20 and noise > 0:
        ratio = stats["resid_std_db"] / noise
        ok_a = 0.5 <= ratio <= 1.8

    print("=== A: CSI layer-1 ===")
    print(
        f"policy={policy_name} seed={seed} slots={log['num_slots']} "
        f"TBs={len(log['cqi_index'])} cqi_delay={stats['cqi_delay_slots']}"
    )
    print(
        f"  corr={stats['corr_aligned']:.4f}  "
        f"resid_std={stats['resid_std_db']:.3f} dB (cfg {noise:g})  "
        f"[{'PASS' if ok_a else 'WARN'}]"
    )
    print(f"  plot -> {plot_path}")

    # --- C: state[0] 테이블 ---
    h_c, rows_c, fail_c = table_c_state0(log)
    _print_table("C: state[0] == cqi/15", h_c, rows_c, table_rows)
    csv_c = out_dir / f"obs_table_C_{tag}.csv"
    md_c = out_dir / f"obs_table_C_{tag}.md"
    _write_csv(csv_c, h_c, rows_c)
    _write_md_table(md_c, f"C: state[0] == cqi/15 ({tag})", h_c, rows_c)
    print(f"  fails={fail_c}/{len(rows_c)}  [{'PASS' if fail_c == 0 else 'FAIL'}]")
    print(f"  table -> {md_c}")
    print(f"  csv   -> {csv_c}")

    # --- D: 히스토리 테이블 ---
    h_d, rows_d, fail_d = table_d_history(log)
    _print_table("D: history (cqi,mcs,ack) aligned (ack_delay=0)", h_d, rows_d, table_rows)
    csv_d = out_dir / f"obs_table_D_{tag}.csv"
    md_d = out_dir / f"obs_table_D_{tag}.md"
    _write_csv(csv_d, h_d, rows_d)
    _write_md_table(md_d, f"D: history aligned ({tag})", h_d, rows_d)
    print(f"  fails={fail_d}/{len(rows_d)}  [{'PASS' if fail_d == 0 else 'FAIL'}]")
    print(f"  table -> {md_d}")
    print(f"  csv   -> {csv_d}")

    ok = ok_a and fail_c == 0 and fail_d == 0
    print("\n" + ("All checks passed." if ok else "Some checks FAILED/WARN."))
    return ok


def main():
    p = argparse.ArgumentParser(description="CSI overlay + C/D state tables (ack_delay=0)")
    p.add_argument("--config", type=Path, default=_ROOT / "configs" / "downlink_la.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-slots", type=int, default=200)
    p.add_argument("--policy", choices=("illa", "olla"), default="illa")
    p.add_argument("--table-rows", type=int, default=15, help="console rows to show")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    ok = run(
        cfg,
        seed=args.seed,
        policy_name=args.policy,
        num_slots=args.num_slots,
        out_dir=args.out_dir,
        table_rows=args.table_rows,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
