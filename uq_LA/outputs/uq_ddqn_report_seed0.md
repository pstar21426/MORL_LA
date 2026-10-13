# UQ-ddqn smoke report (train seed=0, 80 ep)

- Mean eval return (ID): **1033.4** vs OLLA **1017.7** (delta +15.7)
- Epistemic mean ID vs OOD (cqi_bias 3→0): **20.837** vs **19.712**
- Aleatoric mean if num_retx>0 vs ==0: **5.809** vs **1.509** (Spearman vs retx 0.178)
- Epistemic vs 1/sqrt(visit): Spearman **-0.383**

## Interpretation

- **DQN vs DDQN**: paper uses vanilla DQN; DDQN is for stable Q in this LA MDP. Uncertainty head is trained on one-step `r - rho*tau` either way.
- Positive OOD epistemic gap + higher aleatoric on HARQ retries supports disentanglement.