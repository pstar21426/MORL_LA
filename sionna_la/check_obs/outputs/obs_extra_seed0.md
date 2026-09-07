# Extra checks (not in la_baselines_seed0.png)

seed=0  step_up=0.1  step_down=0.9

## OLLA offset vs ACK
- offset step mismatches (ignoring ±20 clip): **0** / 858
- offset range: [-15.50, 0.50] dB
- plot: `obs_olla_offset_seed0.png`
- table: `obs_table_olla_offset_olla_seed0.md`

## HARQ last-tx TBLER vs 1st-tx (retx TBs only)
- plot: `obs_harq_tbler_seed0.png`

## PHY: mean predicted 1st-tx TBLER vs empirical NACK

| policy | mean tbler_first | emp 1st-tx BLER | diff |
| --- | --- | --- | --- |
| ILLA | 0.3218 | 0.3203 | +0.0015 |
| OLLA | 0.1259 | 0.1177 | +0.0082 |

## DDQN
- greedy replay / vs ILLA scatter: `check_obs/check_ddqn.py`
