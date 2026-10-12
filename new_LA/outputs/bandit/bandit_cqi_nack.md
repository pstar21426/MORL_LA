# One-step rate table, variant cqi+nack

Train seeds 1000-1119, eps 0.5->0.05, greedy probe every 10 eps. Final rho 1.088. Eval seeds [205, 206, 207, 208, 209], paired with OLLA.

| policy | return | diff vs OLLA | 95% CI | BLER | MCS | MCS off/on burst | BLER off/on burst |
| --- | ---: | ---: | --- | ---: | ---: | --- | --- |
| OLLA | 1044.4 | +0.0 | [+0.0, +0.0] | 0.110 | 15.1 | 15.2 / 14.1 | 0.079 / 0.426 |
| ILLA | 856.3 | -188.1 | [-232.2, -144.0] | 0.382 | 23.2 | 23.4 / 20.5 | 0.336 / 0.838 |
| cqi+nack | 1107.0 | +62.5 | [+29.3, +95.7] | 0.100 | 11.2 | 11.1 / 11.7 | 0.074 / 0.359 |

Per-seed return: OLLA 1066.5, 942.0, 1044.5, 1153.7, 1015.5; ILLA 862.6, 754.8, 870.5, 918.0, 875.7; cqi+nack 1143.8, 973.0, 1080.4, 1237.7, 1099.9
