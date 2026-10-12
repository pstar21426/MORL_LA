# Score table ± residual vs ILLA / OLLA (train seed=1000, 400 ep)

Queue feature is bits / max TBS. The table score is the action value. The residual is squashed to ±0.05 and is not bootstrapped.

| seed | SCORE ret | RESID ret | ILLA ret | OLLA ret | SCORE MCS | RESID MCS | OLLA MCS | SCORE BLER | RESID BLER | OLLA BLER |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 205 | 1143.8 | 1140.0 | 862.6 | 1066.5 | 10.6 | 10.5 | 15.2 | 0.095 | 0.092 | 0.104 |
| 206 | 995.0 | 985.9 | 754.8 | 942.0 | 10.5 | 10.5 | 13.9 | 0.108 | 0.106 | 0.110 |
| 207 | 1099.0 | 1097.0 | 870.5 | 1044.5 | 10.6 | 10.6 | 13.6 | 0.101 | 0.101 | 0.114 |
| 208 | 1210.6 | 1214.6 | 918.0 | 1153.7 | 10.8 | 10.8 | 15.9 | 0.081 | 0.082 | 0.107 |
| 209 | 1111.6 | 1120.4 | 875.7 | 1015.5 | 10.6 | 10.5 | 17.1 | 0.073 | 0.075 | 0.116 |

**mean±std return** SCORE 1112.0±70.1 · RESID 1111.6±74.2 · ILLA 856.3±54.3 · OLLA 1044.4±68.9

one-step MCS map at nack 0:

- queue 400: `[11, 9, 9, 10, 4, 7, 4, 7, 7, 7, 8, 9, 9, 9, 9, 9]`
- queue 800: `[18, 18, 18, 18, 4, 7, 4, 7, 7, 9, 8, 9, 16, 15, 16, 16]`
- queue 2000: `[26, 28, 28, 19, 4, 7, 4, 7, 7, 9, 8, 9, 16, 15, 16, 25]`
