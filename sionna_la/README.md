# Sionna 5G Downlink Link Adaptation

ACK/NACK은 Sionna `PHYAbstraction` BLER 테이블에서 나옵니다.


| 파일 | 역할 |
|------|------|
| `la_env.py` | `DownlinkLAEnv` — dynamics의 단일 진실 공급원 |
| `channel.py` | AR(1) effective SINR + noisy/delayed CQI |
| `harq.py` | MIESM HARQ-IR 누적 |
| `policies.py` | ILLA / OLLA / ε-greedy |
| `run_la_sim.py` | 롤아웃 → plot + npz |
| `configs/downlink_la.yaml` | 파라미터 |

```python
obs, info = env.reset(seed=0)
a = policy(obs, info)          # ILLA / OLLA / RL
obs, r, term, trunc, info = env.step(a)
```

재전송 슬롯은 `info["is_decision"]=False`이고 action은 무시됩니다.
`run_la_sim.py`는 per-slot npz와 TB 단위 `*_decisions.npz`를 같이 저장합니다.
