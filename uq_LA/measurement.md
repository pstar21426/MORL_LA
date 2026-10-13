## 무엇을 하려고 했나

보고 싶었던 것은 두 가지다.

1. 학습된 정책의 return이 같은 시드의 OLLA보다 높은가.
2. 논문식 집계로 aleatoric과 epistemic이 서로 다른 원인에 반응하는가.

2번은 이 실행에서 성립하지 않았다. 아래 "읽을 수 있는 결론"에 적는다.

MOPO나 oracle 페널티 비교는 하지 않았다. 데이터는 온라인으로 모았고, 실행 정책은 앙상블 평균 Q의 argmax다.

## 이 폴더가 생기기까지의 과정

`new_LA`에서 DDQN이 MCS를 거의 상수로 고르던 이유를 먼저 좁혔다. 슬롯 할인 0.99면 TD 타깃이 수십~백 단위로 커지고, MCS 간 점수 차(대략 0.01~0.1)가 가려졌다. 큐를 버퍼 용량 15000이 아니라 최대 TBS 1688로 나누면 큐 임계가 상태에 남는다. `gamma=0.7`, `queue_feat=tbs`, 학습 보상 `r - rho * tau`에서 DDQN이 OLLA를 앞섰다. 그 설정이 이 폴더 환경의 기본값이다.

그 다음 논문의 불확실성 분리를 붙일 수 있는지 봤다. 첫 구현은 evidential head였다. Q는 TD로 학습하고, 불확실성 head는 bootstrap 없는 one-step `r - rho * tau`에 Normal-Inverse-Gamma를 맞췄다. 재전송 TB에서 aleatoric 엔트로피가 커지기는 했다. `cqi_bias`를 3에서 0으로 바꾼 평가에서는 epistemic이 오히려 줄었다. 분리의 근거로는 약했다.

논문의 네 방법 중 이 환경에 그대로 옮기기 쉬운 것은 ensemble이다. 멤버마다 `(mu, sigma)`가 나와서, aleatoric은 sigma의 평균, epistemic은 mu의 분산으로 논문 본문과 같이 집계할 수 있다. Deep kernel은 분산이 하나라 둘이 안 갈라지고, dropout은 같은 집계가 가능하지만 논문도 데이터에 따라 epistemic이 줄지 않는다고 적는다. Evidential 엔트로피는 보상 단위가 아니라서 나중에 분산을 빼는 비교에 쓰기 어렵다.

그래서 `uq_ensemble.py`와 `train_ensemble.py`를 논문 4절의 ensemble 설명에 맞춰 넣고, evidential 코드는 그대로 두었다. 학습은 `python -u train_ensemble.py --episodes 80`이다. 설정의 `train.episodes`는 120이지만 이 실행은 80이다.

## 환경

`channel.py`, `cqi.py`, `harq.py`, `la_env.py`, `policies.py`는 `new_LA`에서 복사한 것이다. 설정은 `configs/downlink_la.yaml`.

- 에피소드 1000슬롯. 결정은 슬롯마다가 아니라 TB(또는 큐가 비면 idle 슬롯)마다.
- 행동 `a`는 MCS index `a + mcs_min`. 표는 38.214 PDSCH MCS table 1, downlink.
- 상태: 정규화 CQI, CQI/MCS/초전송 ACK 각 3스텝 이력, 큐/최대 TBS.
- SINR 평균 10 dB, AR(1) rho 0.95, innovation 2.5 dB.
- CQI는 지연 3슬롯, 잡음 1.5 dB, bias +3 dB. CQI는 버스트 전 SINR을 본다.
- 디코딩 SINR만 버스트 -8 dB. on/off 평균 20/80슬롯.
- HARQ 재전송 최대 3, 재전송 앞 간격 3슬롯. 그래서 한 TB가 쓰는 슬롯 수 `tau`는 1, 5, 9, 13 근처다.
- 도착 비트는 슬롯당 300~500 균등. 환경이 주는 보상 `r`은 전달 비트/RE다. NACK면 0에 가깝다.

평가 return은 이 환경 보상 `r`의 합이다. 학습 타깃에만 `r - rho * tau`를 쓴다. OLLA와 비교하는 숫자는  shaping 전 return이다.

## 모델과 학습

멤버는 5개다. 각자 같은 구조의 Gaussian Q 네트워크다.

- 몸통: Linear 256, ReLU, Linear 256, ReLU.
- 머리: 행동마다 `mu`와 `log_std`. `log_std`는 -4에서 2로 clamp. 초기 `log_std` 가중치와 bias는 0이라 시작 표준편차는 1이다.
- 멤버 초기화 시드만 다르다. 리플레이는 하나고, 매 업데이트마다 배치 64를 한 번 뽑아 다섯 멤버가 같이 쓴다. Bootstrap 마스크는 없다.
- 멤버마다 타깃 네트워크가 있고 200 step마다 복사한다.

타깃은 논문의 DQN이다. Double DQN이 아니다. 멤버 `k`의 타깃은

```text
y = (r - rho * tau) + gamma^tau * max_a' mu_target_k(s', a')
```

`gamma`는 슬롯당 0.7이다. TB가 `tau`슬롯을 쓰면 할인은 `0.7^tau`다. `rho`는 에피소드 슬롯당 return의 EMA다. 계수 0.1, 시작은 첫 에피소드의 rate.

손실은 고른 행동의 Gaussian NLL이다.

```text
var = exp(2 * log_std)
NLL = 0.5 * (2 * log_std + (y - mu)^2 / var)
```

상수 `log(2π)`는 빠졌다. gradient clip은 5.

행동을 고를 때는 다섯 `mu`의 평균이 최대인 행동을 쓴다. 학습 중 epsilon-greedy. epsilon은 1.0에서 0.01까지 30000 step. 평가와 플롯은 greedy, 즉 epsilon 0이다.

디바이스는 CPU다. `sionna_la`의 PyTorch가 `2.13.0+cpu`라 `cuda.is_available()`이 거짓이다. RTX 3070은 드라이버에는 보인다. 네트워크가 작아서 시간의 대부분은 Sionna 환경 스텝이다. 80에피소드에 약 10분이 걸렸다.

학습 로그의 배치 평균은 아래다. 여기의 `u_alea`, `u_epist`는 그 배치에서 실제로 고른 행동에 대한 값이다.


| ep  | return | rate  | rho   | eps   | NLL   | u_alea | u_epist |
| --- | ------ | ----- | ----- | ----- | ----- | ------ | ------- |
| 1   | 622.3  | 0.622 | 0.622 | 0.990 | 1.626 | 5.701  | 0.141   |
| 20  | 779.8  | 0.780 | 0.791 | 0.767 | 1.027 | 2.676  | 0.308   |
| 40  | 983.9  | 0.984 | 0.909 | 0.494 | 1.242 | 2.685  | 0.123   |
| 60  | 1018.6 | 1.019 | 0.978 | 0.192 | 0.739 | 2.553  | 0.188   |
| 80  | 1004.1 | 1.004 | 1.098 | 0.010 | 0.960 | 2.507  | 0.143   |


학습 중 epistemic은 줄었다가 다시 커진다. "학습이 길어지면 epistemic이 단조롭게 준다"는 논문 desideratum 3.1은 이 로그만으로는 만족하지 않는다.

체크포인트: `outputs/uq_ensemble_k5_seed0.pt`.

## 불확실성을 어떻게 재나

평가에서 결정을 내리기 직전, **그 결정에 고른 행동**에 대해 한 번씩 기록한다. 결과는 아직 보지 않은 시점이다.

논문 4절 집계와 같다.

```text
mu      = (1/K) * sum_k mu_k
u_alea  = (1/K) * sum_k sigma_k
u_epist = (1/K) * sum_k (mu_k - mu)^2
```

`sigma_k = exp(log_std_k)`다. aleatoric은 엔트로피가 아니다. 멤버가 각자 예측한 가우시안 표준편차의 평균이다. epistemic은 다섯 평균 예측이 서로 얼마나 다른지다. 단위는 Q, 즉 `r - rho*tau`에 할인이 섞인 값의 스케일이다. bit/RE로 읽지 않는다.

이 두 값은 정책에 다시 넣지 않았다. MCS는 `mu`만 보고 고른다. 불확실성은 측정용이다.

### 평가 프로토콜

학습 롤아웃 시드는 0부터 79다. 평가는 205, 206, 207이다. 학습 구간과 안 겹친다. 각 시드에서 채널, CQI 잡음, 버스트, HARQ 난수는 `seed_phy(seed)` 후 `env.reset(seed)`라 정책만 다르면 같은 채널을 본다.

세 가지 비교를 했다.

**Return.** greedy 앙상블과 OLLA(ILLA는 플롯에만)를 같은 시드에 1000슬롯 돌리고 환경 보상 합을 비교한다. OLLA는 CQI에서 온 SINR에 offset을 더해 MCS를 고르고, 초전송 ACK면 offset을 +0.1 dB, NACK면 `-0.1*(1/0.1-1)` dB 움직인다.

**Aleatoric과 재전송.** 기록된 결정 중 idle이 아닌 TB만 쓴다. 그 TB의 `num_retx`가 0이면 초전송, 0보다 크면 재전송이 있었던 TB다. 세 시드를 합쳐 TB 1888개 중 재전송은 187개다. 그룹 평균과, `u_alea`와 `num_retx`의 Spearman 순위를 본다. 재전송이 많으면 디코딩이 더 흔들렸다는 사후 표시라서, aleatoric이 그 잡음을 보면 재전송 그룹의 `u_alea`가 커져야 한다. 예측 시점에 재전송 여부를 넣지는 않는다.

**Epistemic과 경험, 그리고 bias를 끈 평가.** 학습 동안 CQI별로 결정을 몇 번 했는지를 `cqi_visits`에 센다. 평가 때의 Spearman은 `u_epist`와 `1/sqrt(visit+1)`이다. visit가 작으면 x가 크다. 상관이 양수면 적게 본 CQI에서 epistemic이 크다.

OOD라고 부른 평가는 재학습이 아니다. 같은 체크포인트로 `cqi_bias_db`만 3에서 0으로 바꾸고 같은 시드를 돈다. 상태가 학습 때와 달라지니 epistemic 평균이 커지면, 논문의 "학습과 다른 환경에서 epistemic이 크다" 쪽에 가깝다. 채널 전체를 모르는 상태로 바꾼 것은 아니다.

추가로, 초전송/재전송 그룹에서 `u_epist` 평균과 두 불확실성의 피어슨 상관을 구했다. 이 숫자는 자동 리포트에는 없고, 분리 여부를 판단할 때 계산했다.

## 나온 숫자

자동 리포트 `outputs/uq_ensemble_k5_report_seed0.md`.


| 항목                                     | 값                                 |
| -------------------------------------- | --------------------------------- |
| 평가 return 평균                           | 앙상블 1096.9, OLLA 1017.7, 차이 +79.2 |
| 시드 205 / 206 / 207 앙상블                 | 1140.7 / 1046.0 / 1103.9          |
| 같은 시드 OLLA                             | 1066.5 / 942.0 / 1044.5           |
| 시드별 차이의 SE                             | 13.1 (플롯 기준, n=3)                 |
| `u_alea` 초전송 / 재전송                     | 1.474 / 2.832                     |
| Spearman(`u_alea`, `num_retx`)         | 0.158                             |
| `u_epist` 초전송 / 재전송                    | 0.029 / 0.070                     |
| 피어슨(`u_alea`, `u_epist`)               | 0.468                             |
| `u_epist` 평균, bias 3 / bias 0          | 0.0327 / 0.0446                   |
| Spearman(`u_epist`, `1/sqrt(visit+1)`) | +0.565                            |


플롯을 다시 돌릴 때의 MCS와 초전송 BLER는 아래다. return은 위와 같다.


| 시드  | 정책   | 평균 MCS | 초전송 BLER |
| --- | ---- | ------ | -------- |
| 205 | ILLA | 24.15  | 0.421    |
| 205 | OLLA | 15.20  | 0.104    |
| 205 | ENS  | 13.15  | 0.085    |
| 206 | ILLA | 21.80  | 0.406    |
| 206 | OLLA | 13.89  | 0.110    |
| 206 | ENS  | 12.82  | 0.102    |
| 207 | ILLA | 22.25  | 0.378    |
| 207 | OLLA | 13.57  | 0.114    |
| 207 | ENS  | 13.20  | 0.111    |


ILLA는 세 시드 모두 return이 가장 낮다. BLER가 0.4 근처인 높은 MCS를 고르기 때문이다.

## 플롯을 어떻게 읽나

`outputs/plots/ensemble_eval_paired.png`

왼쪽 막대는 시드별 에피소드 return이다. 오른쪽은 앙상블 minus OLLA다. 세 시드 모두 양수이고 평균은 +79.2다. 시드가 3개라 SE 13은 넓다. 방향은 맞지만 정밀한 개선 폭으로 쓰지 않는다.

`outputs/plots/ensemble_policy_map.png`

왼쪽은 보고된 CQI에서 실제로 고른 MCS의 평균이다. 세 시드의 TB를 합쳤다. CQI가 오르면 OLLA와 앙상블 모두 MCS를 올린다. 높은 CQI에서 앙상블이 더 낮다. CQI 15 근처에서 OLLA는 MCS 약 22, 앙상블은 약 16이다. 평균 MCS도 앙상블이 약 13, OLLA가 약 14다. return이 높은 이유가 더 높은 MCS를 골라서가 아니다.

가운데와 오른쪽은 최근 3개 초전송 ACK 이력에서 값이 0인 개수로 나눈 것이다. 아직 없는 이력은 -1이라 NACK로 세지 않는다. CQI 칸의 표본이 8개 미만이면 점을 빼 두었다. OLLA 곡선은 NACK 개수가 달라도 거의 겹친다. offset이 누적이라 에피소드 평균에서는 최근 NACK 개수와 MCS가 잘 안 갈라진다. 앙상블은 최근 NACK가 2개 이상이면 같은 CQI에서 MCS를 더 낮다. 표본은 0 NACK 1452, 1 NACK 334, 2개 이상 102다.

`outputs/plots/uq_ensemble_diag.png`

이 그림으로 분리를 판단하지 않는다. 왼쪽 상자는 재전송 여부에 따른 `u_alea`라서 위의 1.47 대 2.83과 같은 비교다. 제목에 entropy라고 적힌 것은 evidential 플롯 스크립트를 그대로 써서 그렇다. 값은 표준편차 평균이다. 오른쪽은 epistemic 대 `1/sqrt(평가 로그 순서)`다. CQI 방문 횟수가 아니다. 뒤쪽 결정일수록 x가 0에 몰려 점이 왼쪽에 쌓인다.

## 읽을 수 있는 결론

정책 성능은 이 세 시드에서 OLLA보다 높다. MCS는 OLLA보다 보수적이고, 최근 NACK가 쌓이면 더 내려간다. BLER는 OLLA와 비슷하거나 조금 낮다.

불확실성 분리는 이 결과로 주장하지 않는다. 분리가 되려면 재전송처럼 줄일 수 없는 흔들림에는 `u_alea`만 움직이고, 덜 본 상태나 학습 때와 다른 bias에는 `u_epist`만 움직여야 한다. 실제로는 재전송 TB에서 둘 다 약 두 배 커졌고, 서로 상관 0.47이다. 위험한 TB에서 두 숫자가 같이 커진 것이다. 적게 본 CQI에서 epistemic이 크다는 Spearman +0.57과, bias 0에서 epistemic이 0.033에서 0.045로 오른 것은 epistemic 쪽 신호로만 남긴다. bias를 끈 차이는 작다.

학습 배치의 epistemic도 에피소드가 늘면서 단조롭게 줄지 않았다.

## 아직 안 한 것

- 버스트, CQI 지연, CQI bias를 하나씩 끄고 **다시 학습**하는 ablation. 평가에서만 끄는 것과 다르다. 앙상블 진단이 분리로 읽히지 않았으니, 그 ablation은 "어느 채널 요소가 불확실성을 만드는가"를 보기 위한 것이다. 기본 설정에서 그 요소를 빼자는 뜻은 아니다.
- vanilla 예측 분산을 페널티로 넣은 정책과, 알려진 ACK 분산을 뺀 oracle 페널티의 return 비교. 그건 MOPO식 오프라인 실험이고 이 실행이 아니다.
- `new_LA` DDQN(gamma 0.7, 더 긴 학습, 시드 205–209)과의 같은 시드 재비교. 이 노트의 +79는 앙상블 80에피소드, 시드 3개의 숫자다.



## 다시 돌리는 명령

환경은 `conda activate sionna_la`. 작업 디렉터리는 `uq_LA`.

```text
python -u train_ensemble.py --episodes 80
python -u plot_ensemble_eval.py
```

두 번째 명령이 체크포인트로 시드 205–207을 다시 굴려 paired return과 MCS 맵을 덮어쓴다.