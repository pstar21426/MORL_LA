# Extra checks (not in la_baselines_seed0.png)

seed=0  step_up=0.1  step_down=0.9

## OLLA offset vs ACK
- offset step mismatches (ignoring ±20 clip): **0** / 855
- offset range: [-16.80, 1.00] dB
- plot: `obs_olla_offset_seed0.png`
- table: `obs_table_olla_offset_olla_seed0.md`

## HARQ last-tx TBLER vs 1st-tx (retx TBs only)
- plot: `obs_harq_tbler_seed0.png`

## PHY: mean predicted 1st-tx TBLER vs empirical NACK

| policy | mean tbler_first | emp 1st-tx BLER | diff |
| --- | --- | --- | --- |
| ILLA | 0.2874 | 0.2792 | +0.0082 |
| OLLA | 0.1260 | 0.1193 | +0.0067 |

## DDQN
- greedy replay / vs ILLA scatter: `check_obs/check_ddqn.py`
