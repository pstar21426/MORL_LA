# Differential Q-learning table (queue bins), variant cqi+nack, fixed, run seed 1

fixed duty. Train seeds 11000-11399, eps 0.5->0.05, greedy probe every 10 eps. Final rho 0.938. Eval seeds [205, 206, 207, 208, 209], paired with OLLA.

| policy | return | diff vs OLLA | 95% CI | BLER | MCS | MCS off/on burst | BLER off/on burst |
| --- | ---: | ---: | --- | ---: | ---: | --- | --- |
| OLLA | 1044.4 | +0.0 | [+0.0, +0.0] | 0.110 | 15.1 | 15.2 / 14.1 | 0.079 / 0.426 |
| ILLA | 856.3 | -188.1 | [-232.2, -144.0] | 0.382 | 23.2 | 23.4 / 20.5 | 0.336 / 0.838 |
| qlearn cqi+nack | 1042.7 | -1.7 | [-35.5, +32.0] | 0.096 | 10.6 | 10.7 / 10.4 | 0.069 / 0.294 |

Per-seed return: OLLA 1066.5, 942.0, 1044.5, 1153.7, 1015.5; ILLA 862.6, 754.8, 870.5, 918.0, 875.7; qlearn cqi+nack 1070.8, 934.7, 1033.5, 1119.2, 1055.3
