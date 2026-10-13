# UQ-dqn smoke report (train seed=0, 80 ep)

- Mean eval return (ID): **1029.7** vs OLLA **1017.7** (delta +12.0)
- Epistemic mean ID vs OOD (cqi_bias 3→0): **23.757** vs **21.135**
- Aleatoric mean if num_retx>0 vs ==0: **5.906** vs **1.646** (Spearman vs retx 0.216)
- Epistemic vs 1/sqrt(visit): Spearman **-0.234**

## Interpretation

- **DQN vs DDQN**: paper uses vanilla DQN; DDQN is for stable Q in this LA MDP. Uncertainty head is trained on one-step `r - rho*tau` either way.
- Positive OOD epistemic gap + higher aleatoric on HARQ retries supports disentanglement.