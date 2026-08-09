# MOPO aleatoric overpessimism (Appendix A toy LA)

ε=0.25, seed=0, λ=5, dyn=1000, policy=3000, eval=1000

| Method | ε=0.25 | notes |
|--------|--------|-------|
| Greedy (behavioral) | 1024.82 | |
| MOPO vanilla (pred reward var) | 953.91 | ~7% below greedy |
| MOPO oracle_p (ACK aleatoric) | 1275.30 | above greedy |

Pattern still not: oracle/vanilla << greedy.
