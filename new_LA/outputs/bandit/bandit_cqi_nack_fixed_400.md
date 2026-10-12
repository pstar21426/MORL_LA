# One-step rate table, variant cqi+nack, fixed

fixed duty. Train seeds 1000-1399, eps 0.5->0.05, greedy probe every 10 eps. Final rho 1.199. Eval seeds [205, 206, 207, 208, 209], paired with OLLA.

| policy | return | diff vs OLLA | 95% CI | BLER | MCS | MCS off/on burst | BLER off/on burst |
| --- | ---: | ---: | --- | ---: | ---: | --- | --- |
| OLLA | 1044.4 | +0.0 | [+0.0, +0.0] | 0.110 | 15.1 | 15.2 / 14.1 | 0.079 / 0.426 |
| ILLA | 856.3 | -188.1 | [-232.2, -144.0] | 0.382 | 23.2 | 23.4 / 20.5 | 0.336 / 0.838 |
| cqi+nack | 1120.4 | +76.0 | [+49.7, +102.2] | 0.079 | 10.2 | 10.2 / 10.2 | 0.055 / 0.304 |

Per-seed return: OLLA 1066.5, 942.0, 1044.5, 1153.7, 1015.5; ILLA 862.6, 754.8, 870.5, 918.0, 875.7; cqi+nack 1152.1, 1000.9, 1097.3, 1230.6, 1121.0
