import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_csv(path: Path, headers: list[str], rows: list[list]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)


def _write_md(path: Path, title: str, headers: list[str], rows: list[list]):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        f"rows: {len(rows)}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def reconstruct_offset(acks, step_up, step_down, lo, hi):
    """ack_delay=0: TB i sees first-ack of TB i-1 before choosing MCS."""
    offset = 0.0
    out = np.empty(len(acks), dtype=np.float64)
    for i, ack in enumerate(acks):
        if i > 0:
            if int(acks[i - 1]) == 1:
                offset += step_up
            else:
                offset -= step_down
            offset = float(np.clip(offset, lo, hi))
        out[i] = offset
    return out


def plot_olla_offset(tb, offset, acks, step_up, step_down, out_path: Path):
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(tb, offset, color="C1", lw=1.1, label="OLLA offset [dB]")
    ax.axhline(0.0, color="k", lw=0.6, alpha=0.4)
    nack = np.asarray(acks) == 0
    if np.any(nack):
        ax.scatter(
            tb[nack],
            offset[nack],
            s=10,
            color="C3",
            zorder=3,
            label="decision after NACK (offset already updated)",
        )
    ax.set_xlabel("TB / decision index")
    ax.set_ylabel("offset [dB]")
    ax.set_title(
        f"OLLA offset (ack_delay=0)  |  "
        f"step_up={step_up:g}  step_down={step_down:g}  "
        f"min={offset.min():.2f}  max={offset.max():.2f}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_harq_tbler(illa, olla, out_path: Path):
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.6))
    for ax, name, res in ((axs[0], "ILLA", illa), (axs[1], "OLLA", olla)):
        retx = np.asarray(res["num_retx"]) > 0
        x = np.asarray(res["tbler"], dtype=float)[retx]
        y = np.asarray(res["tbler_last"], dtype=float)[retx]
        ax.scatter(x, y, s=10, alpha=0.45, color="C0" if name == "ILLA" else "C1")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.7)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("1st-tx TBLER")
        ax.set_ylabel("last-tx TBLER")
        if x.size:
            frac = float(np.mean(y < x - 1e-12))
            ax.set_title(
                f"{name} retx TBs (n={x.size})\n"
                f"last < first: {frac:.0%}  mean Δ={np.mean(x - y):.3f}"
            )
        else:
            ax.set_title(f"{name} (no retx)")
        ax.grid(True, alpha=0.3)
    fig.suptitle("HARQ: TBLER should drop on retx (below y = x)", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=_ROOT / "configs" / "downlink_la.yaml")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--npz-dir", type=Path, default=None)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    npz_dir = args.npz_dir or (_ROOT / cfg.get("out_dir", "outputs"))
    if not npz_dir.is_absolute():
        npz_dir = _ROOT / npz_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    illa_path = npz_dir / f"la_illa_seed{args.seed}.npz"
    olla_path = npz_dir / f"la_olla_seed{args.seed}.npz"
    for path in (illa_path, olla_path):
        if not path.is_file():
            print(f"Missing {path}")
            sys.exit(1)

    illa = dict(np.load(illa_path, allow_pickle=True))
    olla = dict(np.load(olla_path, allow_pickle=True))

    bler_target = float(cfg["bler_target"])
    step_up = float(cfg.get("olla_step_up_db", 0.1))
    step_down = step_up * (1.0 / bler_target - 1.0)
    acks = np.asarray(olla["ack"], dtype=int)
    offset = reconstruct_offset(acks, step_up, step_down, -20.0, 20.0)
    tb = np.arange(len(acks))
    d_off = np.zeros_like(offset)
    d_off[1:] = offset[1:] - offset[:-1]

    plot_olla_offset(
        tb,
        offset,
        acks,
        step_up,
        step_down,
        out_dir / f"obs_olla_offset_seed{args.seed}.png",
    )
    plot_harq_tbler(illa, olla, out_dir / f"obs_harq_tbler_seed{args.seed}.png")

    headers = ["tb", "cqi", "mcs", "ack", "offset_db", "d_offset", "expect_d"]
    rows = []
    n_bad = 0
    for i in range(len(acks)):
        if i == 0:
            expect = 0.0
        elif int(acks[i - 1]) == 1:
            expect = step_up
        else:
            expect = -step_down
        # clip can shrink the step at the rails
        ok = i == 0 or abs(float(d_off[i]) - expect) < 1e-9 or abs(offset[i]) >= 20.0 - 1e-9
        if not ok:
            n_bad += 1
        rows.append(
            [
                i,
                int(olla["cqi_index"][i]),
                int(olla["mcs_used"][i]),
                int(acks[i]),
                f"{offset[i]:.3f}",
                f"{d_off[i]:.3f}",
                f"{expect:.3f}",
            ]
        )
    tag = f"olla_seed{args.seed}"
    _write_md(
        out_dir / f"obs_table_olla_offset_{tag}.md",
        f"OLLA offset from ACK (ack_delay=0, {tag})",
        headers,
        rows,
    )
    _write_csv(out_dir / f"obs_table_olla_offset_{tag}.csv", headers, rows)

    def phy_line(name, res):
        pred = float(np.mean(res["tbler"]))
        emp = float(1.0 - np.asarray(res["ack"]).mean())
        return f"| {name} | {pred:.4f} | {emp:.4f} | {pred - emp:+.4f} |"

    summary = [
        "# Extra checks (not in la_baselines_seed0.png)",
        "",
        f"seed={args.seed}  step_up={step_up:g}  step_down={step_down:g}",
        "",
        "## OLLA offset vs ACK",
        f"- offset step mismatches (ignoring ±20 clip): **{n_bad}** / {len(acks)}",
        f"- offset range: [{offset.min():.2f}, {offset.max():.2f}] dB",
        f"- plot: `obs_olla_offset_seed{args.seed}.png`",
        f"- table: `obs_table_olla_offset_{tag}.md`",
        "",
        "## HARQ last-tx TBLER vs 1st-tx (retx TBs only)",
        f"- plot: `obs_harq_tbler_seed{args.seed}.png`",
        "",
        "## PHY: mean predicted 1st-tx TBLER vs empirical NACK",
        "",
        "| policy | mean tbler_first | emp 1st-tx BLER | diff |",
        "| --- | --- | --- | --- |",
        phy_line("ILLA", illa),
        phy_line("OLLA", olla),
        "",
        "## DDQN",
        "- greedy replay / vs ILLA scatter: `check_obs/check_ddqn.py`",
        "",
    ]
    (out_dir / f"obs_extra_seed{args.seed}.md").write_text("\n".join(summary), encoding="utf-8")

    print(f"OLLA offset mismatches: {n_bad}/{len(acks)}")
    print(f"Wrote {out_dir / f'obs_olla_offset_seed{args.seed}.png'}")
    print(f"Wrote {out_dir / f'obs_harq_tbler_seed{args.seed}.png'}")
    print(f"Wrote {out_dir / f'obs_extra_seed{args.seed}.md'}")
    sys.exit(0 if n_bad == 0 else 1)


if __name__ == "__main__":
    main()
