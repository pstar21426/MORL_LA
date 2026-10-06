# Downlink link adaptation: 문제 정의

Saxena, Tullberg, and Jaldén, “Reinforcement Learning for Efficient and Tuning-Free Link Adaptation,” *IEEE Trans. Wireless Commun.*, 2022의 Section III 전개를 따른다. 그 논문의 전송 처리량은 한 전송의 정규화 비트수와 ACK의 곱이고, 행동은 그 전송의 보상에만 영향을 준다. 아래 모델은 `new_LA` 시뮬레이터와 `configs/downlink_la.yaml`의 현재 설정(`drop_penalty = 0`, `overflow_penalty = 0`)을 기준으로 한다. 시간은 슬롯 $$t = 0,\ldots,T-1$$, $$T = 1000$$ 이다.

## 1. 시스템 모델

### 1.1 채널

진짜 SINR $$\gamma_t$$(dB)는 정상상태에서 시작한 AR(1)을 구간 $$[\gamma_{\min},\gamma_{\max}] = [-5,30]$$ 으로 자른 과정이다.

$$
\tilde\gamma_0 = \mu + \sigma_s z_0, \qquad
\tilde\gamma_{t+1} = \mu + \rho(\tilde\gamma_t - \mu) + \sigma_\varepsilon z_{t+1}, \qquad
\gamma_t = \mathrm{clip}(\tilde\gamma_t,\gamma_{\min},\gamma_{\max}).
\qquad (1)
$$

$$z_t$$ 는 표준정규, $$\mu = 10$$, $$\rho = 0.95$$, $$\sigma_\varepsilon = 2.5$$ 이고 정상 표준편차는 $$\sigma_s = \sigma_\varepsilon/\sqrt{1-\rho^2}$$ 이다.

CQI를 만들기 전의 SINR은 지연, 상수 편향, 측정 잡음을 거친 값이다.

$$
\hat\gamma_t = \mathrm{clip}\!\left(\gamma_{(t-d)_+} + b + \nu_t,\;\gamma_{\min},\gamma_{\max}\right),
\qquad \nu_t \sim \mathcal N(0,\sigma_\nu^2).
\qquad (2)
$$

$$d = 3$$, $$b = +3\,\mathrm{dB}$$, $$\sigma_\nu = 1.5$$ 이고, $$(t-d)_+ = 0$$ 이면 $$\gamma_0$$ 을 쓴다. 디코딩에는 $$\gamma_t$$ 를 쓰고, CQI에는 $$\hat\gamma_t$$ 를 쓴다.

### 1.2 CQI와 MCS

할당 RE 수는 $$N_{\mathrm{RE}} = 300$$ 으로 고정이다. MCS 인덱스는 표 1의 $$m \in \mathcal M = \{3,\ldots,28\}$$ 이다. CQI $$q_t \in \{0,\ldots,15\}$$ 는 목표 블록 오류율 $$\varepsilon = 0.1$$ 이하인 가장 높은 인덱스다.

$$
q_t = \max\left\{ q \in \{0,\ldots,15\} : \mathrm{TBLER}\!\left(\mathrm{MCS}(q),\hat\gamma_t\right) \le \varepsilon \right\}.
\qquad (3)
$$

해당하는 MCS가 없으면 $$q_t = 0$$ 이다. $$\mathrm{MCS}(q)$$ 는 CQI 표의 스펙트럼 효율에 가장 가까운 MCS이고, CQI 0–3은 모두 MCS 3으로 모인다. $$\mathrm{TBLER}$$ 은 Sionna PHY abstraction의 코드블록 BLER로부터 $$1-(1-\mathrm{BLER})^{N_{\mathrm{CB}}}$$ 로 계산한다.

### 1.3 도착과 버퍼

슬롯 $$t$$ 의 시작에 비트 $$A_t$$ 가 균등 도착한다.

$$
A_t \sim \mathrm{Unif}\{A_{\min},\ldots,A_{\max}\}, \qquad A_{\min}=300,\; A_{\max}=500.
\qquad (4)
$$

같은 에피소드 시드에서 $$A_t$$ 의 수열은 정책과 무관하다. 슬롯마다 도착이 정확히 한 번이기 때문이다. 버퍼 용량은 $$Q^{\max} = 15000$$ 이다. 도착 중 용량을 넘는 부분은 들어가지 않으며, 이 비트 수를 오버플로(tail drop)라 한다.

### 1.4 전송과 HARQ

정책은 버퍼가 비어 있지 않을 때 MCS $$m$$ 을 하나 고르고, 그 전송 블록(TB)이 끝날 때까지 다시 고르지 않는다. 실리는 비트 수는 결정 시점의 큐 $$Q$$ 와 코드블록 크기 $$\mathrm{TBS}(m)$$ 의 하한이다.

$$
P = \min\!\big(Q,\;\mathrm{TBS}(m)\big).
\qquad (5)
$$

$$\mathrm{TBS}(m)$$ 은 Sionna의 코드블록 크기와 코드블록 수의 곱이며, 이 설정에서는 TB CRC 16비트를 포함한다. 따라서 아래 비트 등식은 응용 비트보다 이 16비트만큼 넓은 시뮬레이터 비트 기준이다.

초기 전송의 성공 확률은 $$\gamma$$ 에서의 TBLER이다. NACK이면 최대 $$R_{\max} = 3$$ 번까지 재전송한다. 재전송 앞에는 $$g = 3$$ 슬롯을 비우고, 그 슬롯에는 도착만 있다. 재전송 $$n$$ 회의 유효 SINR은 같은 변조에서 상호정보 평균의 역이고, 조회 부호율은 초기 부호율을 전송 횟수 $$n+1$$ 로 나눈 값이다. 초기 전송의 TBS는 유지한다.

$$
\gamma^{\mathrm{eff}}_n = I^{-1}\!\left(\frac{1}{n+1}\sum_{i=0}^{n} I(\gamma_{t_i})\right), \qquad R_n = R_0/(n+1).
\qquad (6)
$$

ACK 또는 재전송 한도를 넘기면 $$P$$ 를 큐에서 뺀다. 한도를 넘긴 경우가 HARQ 폐기이다. 에피소드가 TB 도중에 끝나면 그 $$P$$ 는 큐에 남고, 폐기에도 전달에도 넣지 않는다.

한 TB가 차지하는 슬롯 수는 초기 전송에서 끝나면 1, 재전송 1·2·3회에서 끝나면 각각 5, 9, 13이다.

## 2. 비트 보존과 목적함수

에피소드 전체에서 들어온 비트는 전달, HARQ 폐기, 오버플로, 종료 시 버퍼로만 나뉜다.

$$
\sum_{t=0}^{T-1} A_t
  = B^{\mathrm{del}} + B^{\mathrm{harq}} + B^{\mathrm{ovf}} + Q_T.
\qquad (7)
$$

$$B^{\mathrm{del}}$$ 은 성공한 TB의 $$P$$ 의 합, $$B^{\mathrm{harq}}$$ 는 HARQ로 버린 $$P$$ 의 합, $$B^{\mathrm{ovf}}$$ 는 용량을 넘어 거부된 도착의 합, $$Q_T$$ 는 슬롯 $$T$$ 에 큐에 남은 비트이다. 좌변은 시드가 같으면 정책 사이에서 같다.

링크 적응의 목적은 기대 전달 비트의 최대화이다.

$$
\max_{\pi}\; \mathbb E_{\pi}\!\left[B^{\mathrm{del}}\right].
\qquad (8)
$$

(7)에서 좌변이 정책에 대해 불변이므로 (8)은 세 손실의 합을 최소화하는 문제와 같다.

$$
\min_{\pi}\; \mathbb E_{\pi}\!\left[B^{\mathrm{harq}} + B^{\mathrm{ovf}} + Q_T\right].
\qquad (9)
$$

세 항의 비트당 가중은 같다. HARQ 폐기 100비트, 오버플로 100비트, 종료 시 버퍼 100비트는 (8)에서 같은 손실이다. (9)는 각 항을 따로 최소화하는 문제가 아니다.

1차 전송 BLER과 OLLA의 BLER 목표 $$\varepsilon = 0.1$$ 은 (8)의 항이 아니다. 채널이 맞는지 보는 진단량이다. HARQ 폐기에 3을 곱하거나 오버플로에 1을 곱하는 보상도 (8)의 항이 아니다. 현재 설정은 그 두 계수를 0으로 두어 스텝 보상이 전달 비트만 세게 한다.

### 2.1 시드 201에서의 값

도착 300–500비트, 편향 $$+3$$ dB, 지연 3슬롯인 시드 201에서 (7)의 항은 다음과 같다. 두 정책 모두 $$\sum_t A_t = 399823$$ 이다. 괄호는 $$N_{\mathrm{RE}} = 300$$ 으로 나눈 값이다.

| | $$B^{\mathrm{del}}$$ | $$B^{\mathrm{ovf}}$$ | $$B^{\mathrm{harq}}$$ | $$Q_T$$ |
| --- | ---: | ---: | ---: | ---: |
| ILLA | 281286 (937.6) | 84852 (282.8) | 19376 (64.6) | 14309 |
| OLLA | 351301 (1171.0) | 29493 (98.3) | 5692 (19.0) | 13337 |

OLLA의 전달이 더 크다. 차이의 대부분은 HARQ 폐기가 아니라 오버플로에서 난다. ILLA는 MCS 평균 21.8, 1차 전송 BLER 0.361이고, OLLA는 MCS 평균 15.1, 1차 전송 BLER 0.111이다.

## 3. MDP

결정 시각은 슬롯이 아니라 TB이다. 인덱스를 $$k = 0,1,\ldots,K-1$$ 로 둔다. $$K$$ 는 정책과 채널에 따라 달라진다. $$k$$ 번째 결정이 차지하는 슬롯 수를 $$\tau_k$$ 라 하면 $$\sum_k \tau_k = T$$ 이다.

### 3.1 행동

$$
a_k \in \mathcal A = \{0,1,\ldots,25\}, \qquad m_k = a_k + 3 \in \mathcal M.
\qquad (10)
$$

$$a_k$$ 는 MCS $$m_k$$ 로 이번 TB를 보내는 행동이다. 재전송 MCS는 행동이 아니다. (6)의 규칙으로 정해진다.

### 3.2 보상

성공한 TB의 보상은 전달 비트를 RE 수로 나눈 값이다. 폐기, 오버플로, 에피소드 도중 끊긴 TB의 보상은 0이다.

$$
r_k =
\begin{cases}
P_k / N_{\mathrm{RE}}, & \text{TB } k \text{가 ACK로 끝나면}, \\
0, & \text{그 외}.
\end{cases}
\qquad (11)
$$

할인 없는 보상의 합은 목적 (8)을 RE 수로 나눈 것이다.

$$
\sum_{k=0}^{K-1} r_k = B^{\mathrm{del}} / N_{\mathrm{RE}}.
\qquad (12)
$$

$$N_{\mathrm{RE}}$$ 는 상수이므로 $$\sum_k r_k$$ 를 최대화하는 정책과 $$B^{\mathrm{del}}$$ 을 최대화하는 정책은 같다. 학습기의 할인율 $$\gamma = 0.99$$ 는 (8)에 없다. $$\gamma < 1$$ 이면 긴 HARQ와 짧은 성공이 한 스텝으로 같게 할인되어, (12)와 다른 목적을 최적화한다.

### 3.3 상태

송신기는 $$\gamma_t$$ 를 보지 못한다. 결정 $$k$$ 의 관측은 현재 CQI, 최근 $$L = 3$$ 개 결정의 CQI·MCS·초기 전송 ACK, 그리고 정규화된 큐이다.

$$
s_k =
\Big(
  \tilde q_k,\;
  \tilde q_{k-1},\ldots,\tilde q_{k-L},\;
  \tilde m_{k-1},\ldots,\tilde m_{k-L},\;
  c^{\mathrm{1st}}_{k-1},\ldots,c^{\mathrm{1st}}_{k-L},\;
  Q_k / Q^{\max}
\Big).
\qquad (13)
$$

$$\tilde q = q/15$$, $$\tilde m = (m-3)/25$$ 이고, 아직 없는 과거는 $$-1$$ 이다. $$c^{\mathrm{1st}}\in\{0,1\}$$ 는 그 TB의 초기 전송 ACK이다. 재전송으로 복구된 TB와 HARQ로 폐기된 TB는 둘 다 $$c^{\mathrm{1st}} = 0$$ 이고, 둘 다 큐에서 $$P$$ 를 제거한다. 큐의 변화만으로는 둘을 구분하지 못한다. 상태 차원은 $$2 + 3L = 11$$ 이다.

$$Q_k$$ 는 그 슬롯의 도착이 이미 반영된 큐이다. 정책은 $$s_k$$ 에서 $$a_k$$ 를 고른다.

$$
a_k \sim \pi(\cdot \mid s_k).
\qquad (14)
$$

### 3.4 전이

$$s_k$$ 와 $$a_k$$ 가 정해지면 환경은 다음을 순서대로 수행한다.

1. 슬롯의 $$\gamma$$ 에서 초기 전송을 하고, NACK이면 (6)과 $$g = 3$$ 갭을 반복한다.
2. 갭과 다음 슬롯마다 $$A_t$$ 를 받아 오버플로를 센다.
3. ACK이면 $$P_k$$ 를 전달로, HARQ 한도 초과면 $$P_k$$ 를 폐기로 큐에서 뺀다.
4. 끝난 시점의 CQI, 정규화 MCS, $$c^{\mathrm{1st}}$$ 를 히스토리에 넣고 $$s_{k+1}$$ 을 만든다.

이 전이는 마코프이다. 다만 상태 (13)은 진짜 SINR, 남은 HARQ 여유, TB가 쓴 슬롯 수를 포함하지 않으므로 부분관측이다. 보상 (11)은 이 부분관측 위에서도 (12)를 만족한다.

행동이 이후 보상에 영향을 주는 경로는 세 가지다.

- $$P_k$$ 만큼 큐가 줄거나, 폐기된 비트는 이후 TB에서 다시 보낼 수 없다.
- $$\tau_k > 1$$ 이면 그 슬롯 동안 새 TB를 열지 못하고, 도착은 오버플로가 될 수 있다.
- 히스토리의 CQI·MCS·ACK가 다음 상태의 특징이 된다.

따라서 한 스텝의 기대 보상만 최대화하는 밴드잇 모델은 이 문제의 전이를 담지 못한다. 논문의 multi-armed bandit은 행동이 이후 보상에 영향을 주지 않을 때의 모델이고, 여기의 결정 문제는 유한 구간의 MDP이다.

$$
\max_{\pi}\; \mathbb E_{\pi}\!\left[\sum_{k=0}^{K-1} r_k\right]
  = \max_{\pi}\; \mathbb E_{\pi}\!\left[B^{\mathrm{del}}\right] / N_{\mathrm{RE}}.
\qquad (15)
$$

## 4. 오프라인 RL

(15)를 환경과 더 상호작용하지 않고, 이미 모은 전이로 푸는 문제를 적는다. 알고리즘을 여기서 구현하지는 않는다.

### 4.1 데이터

행동 정책 $$\pi_\beta$$ 로 에피소드를 굴려 결정 스텝의 전이를 모은다.

$$
\mathcal D = \left\{(s_k, a_k, r_k, s_{k+1})\right\}.
\qquad (16)
$$

$$\pi_\beta$$ 는 ILLA와 OLLA이다. 필요하면 그 위에 $$\varepsilon$$-greedy를 얹어, 두 정책이 잘 고르지 않는 $$\mathcal M$$ 안의 MCS를 채운다. ILLA는 CQI만 보고 MCS를 고르므로, 편향 $$b = +3\,\mathrm{dB}$$ 아래에서 높은 MCS와 높은 큐에 전이가 몰린다. OLLA는 1차 전송 ACK로 오프셋을 걸어 MCS를 낮추고 큐를 비운다. 한 정책만 쓰면 $$s$$ 의 큐 축과 $$a$$ 의 낮은 MCS가 비므로, 두 정책의 데이터를 합친다.

학습에 쓴 시드의 궤적은 평가에 넣지 않는다. 학습 시드가 $$[0, N_{\mathrm{ep}})$$ 이면 평가 시드는 그 구간 밖으로 민다.

### 4.2 학습 목표와 평가

오프라인 학습의 목표는 (15)를 $$\mathcal D$$ 위에서 근사하는 정책 $$\pi$$ 이다. 보유한 평가 시드에서 보고할 양은 $$B^{\mathrm{del}}$$, 또는 (9)의 세 손실 합이다. 1차 전송 BLER은 같이 적되 선택 기준은 아니다.

시드 201의 행동 정책만 비교해도 OLLA의 $$B^{\mathrm{del}}$$ 이 ILLA보다 크다(1171.0 대 937.6 bit/RE, 절 2.1). 오프라인 정책이 이 데이터에서 얻어야 하는 이득은, OLLA가 이미 잡는 편향 보정에 더해, 큐가 가득 찰 때 오버플로를 줄이는 MCS 선택이다.

### 4.3 알고리즘

$$\mathcal D$$ 의 상태-행동 분포와 $$\pi$$ 의 분포가 다르면, $$Q$$ 를 데이터 밖에서 평가할 때 전달량이 부풀려진다. 큐가 높은 상태와 공격적인 MCS가 그 경우에 해당한다. 상태 분포의 이동에 패널티를 주는 오프라인 행동-가치 방법(CQL, IQL)이 이 불일치에 직접 대응한다.

모델 기반 방법(MOPO)은 학습된 전이로 롤아웃을 늘린다. 이 환경에서 전이 오차는 큐와 HARQ 길이로 바로 이어지고, 그 오차가 (11)의 전달 비트로 샌다. 모델 롤아웃은 행동-가치 학습의 보조로 두는 편이 맞다.

### 4.4 열린 선택

- 할인. (12)는 $$\gamma = 1$$ 이다. TB 길이가 1에서 13슬롯까지 변하므로, 슬롯 단위 할인 $$\gamma^{\tau_k}$$ 를 쓰면 긴 HARQ의 이후 보상이 더 깎인다. 문제 (8)은 그 할인을 요구하지 않는다.
- 커버리지. 행동 데이터가 높은 MCS와 높은 큐에 치우치면, 빈 큐에서 낮은 MCS를 고르는 정책을 데이터만으로 확인하기 어렵다. $$\varepsilon$$-greedy의 역할은 이 구멍을 메우는 것이다.
- 부분관측. $$c^{\mathrm{1st}} = 0$$ 인 다음 상태는 재전송 성공과 폐기를 한 상태로 묶는다. 보상 (11)은 둘을 구분하지만, 오프라인 $$Q$$ 는 그 구분을 상태 밖에서 평균한다.
