# One-step rate table, variant cqi

Train seeds 1000-1119, eps 0.5->0.05, greedy probe every 10 eps. Final rho 1.095. Eval seeds [205, 206, 207, 208, 209], paired with OLLA.

| policy | return | diff vs OLLA | 95% CI | BLER | MCS | MCS off/on burst | BLER off/on burst |
| --- | ---: | ---: | --- | ---: | ---: | --- | --- |
| OLLA | 1044.4 | +0.0 | [+0.0, +0.0] | 0.110 | 15.1 | 15.2 / 14.1 | 0.079 / 0.426 |
| ILLA | 856.3 | -188.1 | [-232.2, -144.0] | 0.382 | 23.2 | 23.4 / 20.5 | 0.336 / 0.838 |
| cqi | 1117.2 | +72.8 | [+37.6, +108.0] | 0.088 | 10.6 | 10.5 / 11.7 | 0.058 / 0.445 |

Per-seed return: OLLA 1066.5, 942.0, 1044.5, 1153.7, 1015.5; ILLA 862.6, 754.8, 870.5, 918.0, 875.7; cqi 1149.9, 1009.3, 1071.2, 1239.6, 1116.3
