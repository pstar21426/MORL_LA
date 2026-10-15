# MCS bin counts. Pooled over 3 training seeds and eval seeds 205-209.
Epsilon rows use epsilon=0.01. Alea rows sample N(mean mu, (mean sigma)^2).
mean_sigma is the unweighted mean of the per-MCS alea-ensemble sigma in that bin.
That sigma is averaged on states visited by the alea behavior.

| policy | bin | MCS | count | n_tb | rate | mean_sigma |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| alea | low 3-11 | 3-11 | 800 | 5530 | 0.144665 | 1.2305 |
| alea | mid 12-20 | 12-20 | 1626 | 5530 | 0.294033 | 3.0364 |
| alea | high 21-28 | 21-28 | 3104 | 5530 | 0.561302 | 4.6352 |
| epsilon | low 3-11 | 3-11 | 4292 | 9293 | 0.461853 | 1.2305 |
| epsilon | mid 12-20 | 12-20 | 3804 | 9293 | 0.409340 | 3.0364 |
| epsilon | high 21-28 | 21-28 | 1197 | 9293 | 0.128807 | 4.6352 |
| epist | low 3-11 | 3-11 | 5373 | 9569 | 0.561501 | 1.2305 |
| epist | mid 12-20 | 12-20 | 3139 | 9569 | 0.328038 | 3.0364 |
| epist | high 21-28 | 21-28 | 1057 | 9569 | 0.110461 | 4.6352 |
