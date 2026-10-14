# 온라인 학습에서 aleatoric 샘플링과 epistemic 샘플링

보고하는 결과는 `outputs/des32/curves.png`의 왼쪽이다. 같은 앙상블을 80에피소드 온라인으로 학습할 때, 행동을 aleatoric 분포에서 뽑아 고르면 학습 중 반환이 epistemic 샘플링보다 낮게 유지된다. 논문은 Charpentier, Senanayake, Kochenderfer, Günnemann, arXiv:2206.01558, Desideratum 3.2이다.

## 1. MDP

환경은 `configs/downlink_la.yaml`이다. `channel.py`, `cqi.py`, `harq.py`, `la_env.py`, `policies.py`는 `new_LA`에서 복사했다.

시간은 슬롯이고 에피소드는 1000슬롯이다. 결정은 슬롯마다가 아니라 전송 블록이다. 큐가 비어 있으면 그 결정은 1슬롯 공회전이다.

보고하는 값은 환경 보상 `r`의 합이다. ACK로 끝나면 `r = P/300`이고, 아니면 0이다. `P = min(Q, TBS(m))`이고 300은 할당 RE 수다. HARQ 폐기와 버퍼 넘침의 페널티는 0이다.

관측은 11차원이다. 현재 CQI를 15로 나눈 값, 최근 3개 결정의 CQI·MCS·초전송 ACK, 큐를 최대 TBS 1688로 나눈 값이다. 없는 과거는 -1이고 큐 특징은 1을 넘지 않는다. 디코딩 SINR과 버스트 비트는 상태에 없다.

행동은 MCS `m`이고 범위는 3부터 28까지다. 재전송 MCS는 행동이 아니다. 초전송이 NACK이면 같은 MCS로 최대 3번 더 보내고, 재전송 앞에는 3슬롯을 비운다. 한 블록의 슬롯 수 `tau`는 1, 5, 9, 13 중 하나다. ACK이거나 재전송 한도를 넘기면 `P`를 큐에서 뺀다.

채널은 행동을 바꾸지 않는다. 디코딩 SINR은 평균 10 dB, `rho = 0.95`, innovation 2.5 dB인 AR(1)이다. 버스트가 켜지면 디코딩 SINR만 8 dB 내린다. 켜짐·꺼짐 평균 길이는 20슬롯, 80슬롯이다. CQI는 지연 3슬롯, 편향 +3 dB, 잡음 표준편차 1.5 dB를 거친 손상 전 SINR로 만든다. 슬롯마다 300–500비트가 균등 도착하고 버퍼 용량은 15000이다.

학습 타깃만 `r - rho * tau`를 쓴다. `rho`는 에피소드 슬롯당 반환의 지수평균이고 계수는 0.1이다. 슬롯 할인은 `gamma = 0.7`이라 백업은 `gamma^tau`다. 평가와 아래 곡선의 반환은 shaping 전 `r`의 합이다.

이 설정에 오기 전에, 슬롯 할인 0.99에서는 TD 타깃이 커서 MCS 간 점수 차가 가려졌고 정책이 MCS를 거의 상수로 골랐다. 큐를 용량 15000이 아니라 최대 TBS로 나누고 `gamma = 0.7`, 보상 `r - rho * tau`로 두자 DDQN이 MCS를 CQI에 따라 고르기 시작했다. 그 값이 이 폴더의 기본이다.

## 2. 불확실성을 나누는 방법

논문 4절의 ensemble을 쓴다. 멤버는 5개다. 각자 Gaussian Q 네트워크이고, 몸통은 Linear 256, ReLU, Linear 256, ReLU이며 행동마다 `mu`와 `log_sigma`를 낸다. `log_sigma`는 -4에서 2로 자른다. 멤버 초기화 시드만 다르고 리플레이는 하나다. 배치 64를 한 번 뽑아 다섯 멤버가 같이 쓴다. Bootstrap 마스크는 없다. 타깃 네트워크는 멤버마다 있고 200 step마다 복사한다. Double DQN이 아니다.

멤버 `k`의 타깃은 다음과 같다.

```text
y = (r - rho * tau) + gamma^tau * max_a' mu_target_k(s', a')
```

손실은 고른 행동의 Gaussian NLL이다.

```text
NLL = 0.5 * (2 * log_sigma + (y - mu)^2 / sigma^2)
```

`log(2*pi)`는 넣지 않는다. gradient clip은 5다.

결정 시점의 집계는 논문과 같다.

```text
mu      = (1/K) * sum_k mu_k
u_alea  = (1/K) * sum_k sigma_k
u_epist = (1/K) * sum_k (mu_k - mu)^2
```

`u_alea`는 그 상태와 행동에서 보상이 얼마나 흩어지는지다. 버스트와 HARQ 길이처럼 더 방문해도 사라지지 않는 흔들림이 여기에 해당한다. `u_epist`는 멤버들의 `mu`가 서로 다른 정도다. 그 상태·행동을 모델이 아직 합의하지 못했다는 뜻이다. 단위는 `r - rho * tau`에 할인이 섞인 Q의 스케일이고 bit/RE가 아니다.

평균 `mu`만으로 argmax하는 정책은 이 둘을 행동에 넣지 않는다. 그래서 그 정책의 반환이 높아도, 둘을 구분해서 탐색했기 때문이라고 쓸 수 없다. 재전송이 있었던 블록에서 두 값이 같이 커지는지를 사후에 나누는 것도 행동이 아니다. 재전송은 그 MCS를 고른 뒤의 결과이고, `num_retx > 0`인 블록을 재전송 TB라고 부른다.

논문의 다른 구현은 쓰지 않았다. Deep kernel은 분산이 하나라 둘이 갈라지지 않는다. Dropout은 같은 집계가 가능하지만 논문 Table 1에서 데이터가 늘어도 epistemic이 줄어든다는 항목을 통과하지 못한다. Evidential head는 재전송 TB에서 aleatoric 엔트로피가 커졌으나, CQI 편향을 3에서 0으로 바꾼 평가에서 epistemic이 줄었다.

## 3. Desideratum 3.2

논문 3절의 문장은 이렇다. 다른 조건을 같게 두면, 훈련 중 sampling-aleatoric으로 행동을 고르는 에이전트는 sampling-epistemic보다 샘플 효율이 낮다. 훈련 시간은 환경과 상호작용하며 갱신하는 온라인 학습이다. 테스트 시간의 일반화(3.4)나, 모아 둔 전이만으로 배우는 오프라인 학습은 이 비교가 아니다.

두 규칙은 epsilon-greedy가 없다. 네트워크, 손실, 에피소드 수, 환경 시드 0–79는 같다. 멤버 초기 시드도 같다.

- Aleatoric. 행동마다 `y_a ~ N(mu_a, sigma_a^2)`를 하나 뽑고 `y_a`가 최대인 MCS를 고른다. `mu_a`는 다섯 `mu`의 평균, `sigma_a`는 다섯 `sigma`의 평균이다.
- Epistemic. 멤버 하나를 균등하게 뽑고, 그 멤버의 `mu`만으로 argmax를 한다. Thompson sampling이다. `sigma`는 선택에 넣지 않는다.

학습 중 반환은 그때 실제로 고른 행동의 `r` 합이다. Aleatoric 쪽에는 매 결정의 `sigma` 샘플이 들어 있다. 탐욕 반환은 그 시점의 다섯 `mu` 평균으로만 고른 값이다. 시드 205에서 20에피소드마다 재고, 마지막에 205–207을 OLLA와 짝지어 잰다. 스크립트는 `train_des32.py`이고 결과는 `outputs/des32/`이다.

## 4. 왼쪽 곡선

`outputs/des32/curves.png` 왼쪽이 3.2의 샘플 효율이다. 얇은 선은 에피소드 반환, 굵은 선은 10에피소드 평균이다.

약 15에피소드 이후 epistemic의 이동 평균은 1100 근처에 머물고 aleatoric은 860 근처에 머문다. 80에피소드까지 그 간격이 닫히지 않는다. 마지막 10에피소드 평균은 aleatoric 862.4, epistemic 1129.9다.

이 간격이 말하는 것은 학습 중에 받은 반환이다. `sigma`를 점수에 넣는 탐색이 매 스텝 들어가므로, 배운 Q가 더 나쁘다는 뜻까지는 아니다. 학습 시드는 하나라 곡선의 반복 표준오차는 없다.

오른쪽과 마지막 탐욕 평가는 그 한계를 숫자로 둔다. 시드 205에서 epistemic의 탐욕 반환은 20에피소드에 이미 1153이고, aleatoric은 982로 그 시드 OLLA 1018보다 아래였다가 80에피소드에 1112까지 올라온다. 시드 205–207의 탐욕 평균은 aleatoric 1081.0, epistemic 1098.7, OLLA 1017.7이다. OLLA와의 차이는 +63.3 (SE 10.7)과 +81.1 (SE 9.2)이다. 같은 시드에서 epistemic이 aleatoric보다 높은 양은 평균 17.8, 표준오차 14.5라 0을 포함한다.

따라서 이 그림으로 쓰는 문장은 하나다. 온라인 학습 동안 aleatoric 샘플로 MCS를 고르면, epistemic 샘플로 고를 때보다 받은 반환이 낮게 유지된다. 80에피소드 뒤에 평균 `mu`로 고른 정책이 더 나쁘다는 문장은 쓰지 않는다.

## 다시 돌리는 명령

환경은 `conda activate sionna_la`. 작업 디렉터리는 `uq_LA`.

```text
python -u train_des32.py
```

80에피소드 두 번이다. 그림은 `outputs/des32/curves.png`, 표는 `outputs/des32/summary.md`에 덮어쓴다.
