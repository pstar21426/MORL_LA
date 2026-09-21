# MOPO lambda sweep (unit-mean penalty norm)

Same λ is comparable across modes after normalizing u to mean 1 on real data.

|    Method    |      λ=1      |       λ=5       |       λ=10      |     λ=20     |
|---------------|---------------|----------------|----------------|----------------|
| Greedy (DP) | 1290.97 ± 0.00 | 1290.97 ± 0.00 | 1290.97 ± 0.00 | 1290.97 ± 0.00 |
| MOPO vanilla | -1496.31 ± 0.00 | -1021.03 ± 0.00 | -127.29 ± 0.00 | 712.64 ± 0.00 |
| MOPO epistemic | -806.66 ± 0.00 | 682.74 ± 0.00 | 931.21 ± 0.00 | 1115.32 ± 0.00 |

<penalty term>
vanilla: max_i \sigma_i
epistemic: std_i(\mu_i)