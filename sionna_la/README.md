# Sionna 5G Downlink Link Adaptation (standalone)

기존 `env/la_env.py` (Appendix toy)와 **완전히 분리**된 학습용 폴더입니다.
목표는 Sionna LA를 이해하고, 나중에 MDP → Gym으로 옮기는 것입니다.

## Setup

```bash
conda activate sionna_la
cd sionna_la
python run_la_sim.py
```

Windows에서 RT(ray tracing)는 빼 두고 `sionna-no-rt`만 설치했습니다.
LA + PHYAbstraction 학습에는 충분합니다.

## What runs

1. `channel.py` — 시간 상관 effective SINR [dB] 생성 + noisy CQI
2. Sionna `InnerLoopLinkAdaptation` / `OuterLoopLinkAdaptation` 이 MCS 선택
3. Sionna `PHYAbstraction` 이 true SINR + MCS → ACK/NACK, TBLER, decoded bits
4. `outputs/` 에 plot + npz 저장

## Map to MDP (나중에 Gym화할 때)

| 시뮬 양 | MDP |
|--------|-----|
| noisy CQI `sinr_fb_db` | observation \(o_t\) |
| MCS index | action \(a_t\) |
| PHYAbstraction ACK | Bernoulli transition outcome |
| SE if ACK else 0 | reward \(r_t\) (초안) |
| AR(1) SINR | channel dynamics (나중에 OFDM/RT로 교체) |

ILLA/OLLA는 baseline policy \(\pi\)입니다. RL 에이전트는 같은 자리에서 MCS를 고르면 됩니다.
