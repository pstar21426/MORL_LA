# MOPO lambda sweep (unit-mean penalty norm)

Same λ is comparable across modes after normalizing u to mean 1 on real data.

| Method | λ=1 | λ=5 | λ=10 | λ=20 |
|--------|--------|--------|--------|--------|
| Greedy (eps=0 DP) | 1290.97 ± 0.00 | 1290.97 ± 0.00 | 1290.97 ± 0.00 | 1290.97 ± 0.00 |
| MOPO vanilla (pred var) | 1031.93 ± 0.00 | 1233.10 ± 0.00 | 1299.33 ± 0.00 | 1292.13 ± 0.00 |
| MOPO epistemic only | 1039.05 ± 0.00 | 1294.93 ± 0.00 | 1310.84 ± 0.00 | 1310.90 ± 0.00 |
