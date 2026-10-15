# MCS bin counts. Pooled over 3 training seeds and eval seeds 205-209.
Epsilon rows use epsilon=0.01. Alea rows sample N(mean mu, (mean sigma)^2).
mean_sigma is the unweighted mean of the per-MCS alea-ensemble sigma in that bin.
That sigma is averaged on states visited by the alea behavior.

| policy | bin | MCS | count | n_tb | rate | mean_sigma |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| alea | low 3-11 | 3-11 | 2390 | 6882 | 0.347283 | 3.8701 |
| alea | mid 12-20 | 12-20 | 2330 | 6882 | 0.338564 | 5.0051 |
| alea | high 21-28 | 21-28 | 2162 | 6882 | 0.314153 | 6.1537 |
| epsilon | low 3-11 | 3-11 | 5258 | 9414 | 0.558530 | 3.8701 |
| epsilon | mid 12-20 | 12-20 | 3507 | 9414 | 0.372530 | 5.0051 |
| epsilon | high 21-28 | 21-28 | 649 | 9414 | 0.068940 | 6.1537 |
| epist | low 3-11 | 3-11 | 3831 | 9848 | 0.389013 | 3.8701 |
| epist | mid 12-20 | 12-20 | 5316 | 9848 | 0.539805 | 5.0051 |
| epist | high 21-28 | 21-28 | 701 | 9848 | 0.071182 | 6.1537 |
