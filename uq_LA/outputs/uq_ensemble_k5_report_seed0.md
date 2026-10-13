# Ensemble DQN (K=5, train seed=0, 80 ep)

Paper aggregation: u_alea = mean_k sigma_k, u_epist = mean_k (mu_k - mu)^2.
Each member is a Gaussian Q-network trained with NLL on the DQN target.
The executed policy is argmax of the mean mu. Reward is r - rho * tau, gamma=0.7.

- Mean eval return (ID): **1096.9** vs OLLA **1017.7** (delta +79.2)
- Epistemic mean ID vs OOD (cqi_bias 3→0): **0.0327** vs **0.0446**
- Aleatoric mean if num_retx>0 vs ==0: **2.8317** vs **1.4739** (Spearman vs retx 0.158)
- Epistemic vs 1/sqrt(visit): Spearman **0.565**
