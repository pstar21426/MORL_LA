# Score table ± residual vs ILLA / OLLA (train seed=1000, 2 ep)

Queue feature is bits / max TBS. The table score is the action value. The residual is squashed to ±0.05 and is not bootstrapped.

| seed | SCORE ret | RESID ret | ILLA ret | OLLA ret | SCORE MCS | RESID MCS | OLLA MCS | SCORE BLER | RESID BLER | OLLA BLER |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 205 | 880.2 | 893.2 | 862.6 | 1066.5 | 15.4 | 15.5 | 15.2 | 0.174 | 0.165 | 0.104 |

**mean±std return** SCORE 880.2±0.0 · RESID 893.2±0.0 · ILLA 862.6±0.0 · OLLA 1066.5±0.0

one-step MCS map at nack 0:

- queue 400: `[9, 9, 9, 9, 9, 9, 9, 9, 11, 9, 8, 3, 18, 9, 22, 10]`
- queue 800: `[18, 18, 18, 18, 18, 18, 18, 18, 11, 18, 8, 3, 18, 24, 22, 23]`
- queue 2000: `[28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 8, 28, 28, 24, 28, 23]`
