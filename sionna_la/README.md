# Sionna 5G Downlink Link Adaptation

기존 Appendix toy (`env/la_env.py`)와 분리된 Gymnasium 환경입니다.
ACK/NACK은 Sionna `PHYAbstraction` BLER 테이블에서 나옵니다.

```bash
conda activate sionna_la
cd sionna_la
python run_la_sim.py
python train_dqn.py --episodes 40 --seed 0
python check_obs/check_obs.py --both-delays
python check_obs/check_obs.py --dqn --checkpoint outputs/dqn_seed0.pt
```

| 파일 | 역할 |
|------|------|
| `la_env.py` | `DownlinkLAEnv` — dynamics의 단일 진실 공급원 |
| `channel.py` | AR(1) effective SINR + noisy/delayed CQI |
| `harq.py` | MIESM HARQ-IR 누적 |
| `policies.py` | ILLA (CQI→MCS) / OLLA (CQI→SNR+offset→MCS) / ε-greedy |
| `run_la_sim.py` | ILLA/OLLA 롤아웃 → plot + npz |
| `train_dqn.py` | 간단 DQN 학습 + ILLA/OLLA 비교 eval |
| `plot_t_return.py` | slot vs cum return (ILLA/OLLA/DQN, 에피소드 1판) |
| `plot_dqn_train.py` | DQN 학습 곡선 (episode vs return) |
| `check_obs/check_obs.py` | 관측/state 검증 (invariant + scatter + ILLA oracle); `--dqn`으로 greedy/ILLA 비교 |
| `configs/downlink_la.yaml` | 파라미터 |

```python
state, info = env.reset(seed=0)
a = policy(state, info)          # ILLA / OLLA / RL
state, r, term, trunc, info = env.step(a)
```

State (L=`state_num_lags`, default 3): `[cqi×L, m×L, b×L]`  
- `cqi`: 현재 1 + 과거 decision CQI L−1 (norm)  
- `m` / `b`: 과거 MCS (norm) / first-tx ACK (각 L개)  
`delta_tau`는 `info`에만 있음.  
True SINR은 `info["sinr_true_db"]`.  
TB 로그는 `info["outcome"]` (HARQ drop은 `dropped`, 에피소드 끝 mid-TB는 `truncated_mid_tb`).

