"""
논문 Appendix B의 단순 Link Adaptation (LA) Gym 환경.

실제 5G 시뮬레이터 대신, 다음만 남긴 장난감 MDP:
  - 패킷을 몇 번 재전송했는지 (state)
  - MCS를 얼마나 공격적으로 고를지 (action)
  - 지금 채널이 좋은지 (context x)
  - ACK/NACK 확률 + reward + 다음 상태
"""

from __future__ import annotations  # 타입 힌트를 문자열처럼 늦게 평가 (전방참조 허용)

from typing import Any, Optional, SupportsFloat, Tuple

import numpy as np

# RL 표준 인터페이스: Gymnasium 권장 (구버전 gym 대신)
try:
    import gymnasium as gym
    from gymnasium import spaces  # Discrete / Box 등 행동·관측 공간 정의용
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "This environment requires gymnasium. "
        "Install it with: pip install gymnasium"
    ) from exc


# ---------- Appendix B 기본 하이퍼파라미터 ----------
N_STATES = 5          # 전송 시도 인덱스 k = 0,1,2,3,4  (최대 5번 시도)
N_ACTIONS = 28        # MCS 비슷한 이산 action 개수; 논문에서는 a = 1..28
CONTEXT_HIGH = 500    # 채널 context x ∈ {0,...,499}; 클수록 채널 좋음
SCALE_ACTION = 18.0   # 성공확률 식 p=tanh(x/(18*a)) 의 18
DEFAULT_BETA = 0.5    # 실패 패널티 스케일 β (논문이 고정값을 안 줌 → 조절 가능)
# Appendix A Table A.1 greedy ≈12306; with β=0.5 / 10k steps unscaled ≈2239 → scale≈5.5
DEFAULT_REWARD_SCALE = 5.495217
DEFAULT_MAX_STEPS = 10_000  # 에피소드 길이 상한 (자연 종료 없는 continuous 태스크)


class LAEnv(gym.Env):
    """
    실제 LA 직관:
      - action 크면: 성공 시 데이터 많이 보냄 (보상↑) but 실패 확률↑
      - action 작으면: 안전하게 붙지만 보상 작음
      - 실패하면 재전송 카운트 k 증가, 더 큰 패널티
    """

    metadata = {"render_modes": []}  # 시각화 안 씀 (Gymnasium 메타 필드)

    def __init__(
        self,
        n_states: int = N_STATES,              # 재전송 상태 개수
        n_actions: int = N_ACTIONS,            # MCS 선택지 개수
        context_high: int = CONTEXT_HIGH,      # context 상한 (exclusive): 0..high-1
        beta: float = DEFAULT_BETA,            # 실패 reward 가중치 −β(k+1)
        reward_scale: float = DEFAULT_REWARD_SCALE,  # 전체 reward 배율 (정책 invariant)
        max_episode_steps: int = DEFAULT_MAX_STEPS,  # 이 스텝 지나면 truncated=True
        include_context: bool = True,          # True: obs에 채널 x 포함 / False: k만 (부분관측)
        normalize_obs: bool = False,           # True: obs를 대략 [0,1]로 나눔
        seed: Optional[int] = None,            # 난수 시드 (재현성)
    ) -> None:
        super().__init__()  # gym.Env 초기화

        # ----- 입력 검사 -----
        if n_states < 1:
            raise ValueError("n_states must be >= 1")
        if n_actions < 1:
            raise ValueError("n_actions must be >= 1")
        if context_high < 1:
            raise ValueError("context_high must be >= 1")
        if reward_scale <= 0:
            raise ValueError("reward_scale must be > 0")

        # ----- 설정 저장 -----
        self.n_states = int(n_states)                    # k 범위: 0 .. n_states-1
        self.n_actions = int(n_actions)                  # gym action: 0 .. n_actions-1
        self.context_high = int(context_high)            # x 범위: 0 .. context_high-1
        self.beta = float(beta)                          # 실패 패널티 계수
        self.reward_scale = float(reward_scale)          # return 스케일 (DP argmax 불변)
        self.max_episode_steps = int(max_episode_steps)  # truncate 기준
        self.include_context = bool(include_context)     # 관측에 x를 넣을지
        self.normalize_obs = bool(normalize_obs)         # 관측 정규화 여부

        # action_space: agent가 고를 수 있는 행동 집합
        # Discrete(28) → 정수 0,1,...,27
        # 논문 action은 1..28 이므로 step 안에서 +1 해서 매핑
        self.action_space = spaces.Discrete(self.n_actions)

        # observation_space: agent가 받는 관측 벡터의 형태/범위
        if self.include_context:
            # obs = [재전송횟수 k, 채널 context x]
            low = np.array([0.0, 0.0], dtype=np.float32)  # 각 차원 하한
            high = np.array(
                [float(self.n_states - 1), float(self.context_high - 1)],
                dtype=np.float32,
            )  # 각 차원 상한
        else:
            # 부분 관측: 채널 x를 숨기고 k만 줌 (POMDP 스타일)
            low = np.array([0.0], dtype=np.float32)
            high = np.array([float(self.n_states - 1)], dtype=np.float32)

        # Box: 연속 벡터 관측 (여기선 정수지만 float 벡터로 취급)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self._rng = np.random.default_rng(seed)  # 성공 샘플링 / context 샘플링용 RNG
        self.state: int = 0                      # 현재 전송 시도 인덱스 k
        self.context: int = 0                   # 현재 채널 quality x
        self._step_count: int = 0               # 이번 에피소드에서 step 몇 번 했는지

    # ==================================================================
    # MDP 핵심 함수들 (step 안에서도 쓰고, 나중에 DP 최적정책 계산에도 재사용)
    # ==================================================================

    @staticmethod
    def success_probability(context: int, action: int) -> float:
        """
        논문 식: p_success = tanh( x / (18 · a) )
          - context(x) 크면 → 채널 좋음 → 성공 확률↑
          - action(a) 크면 → aggressive MCS → 성공 확률↓
        action은 논문 인덱스(1..28) 기준.
        """
        if action < 1:
            # gym action(0-based)을 그대로 넣으면 버그 → paper 인덱스를 쓰라는 뜻
            raise ValueError("action must be in 1..n_actions (paper indexing)")
        # tanh: 결과를 (0,1) 근처로 부드럽게 눌러 확률처럼 씀
        return float(np.tanh(context / (SCALE_ACTION * action)))

    @staticmethod
    def aleatoric_uncertainty(context: int, action: int) -> float:
        """
        Oracle aleatoric uncertainty from ACK/NACK Bernoulli:
          u = p (1 - p),  p = success_probability(x, a)
        """
        p = LAEnv.success_probability(context, action)
        return float(p * (1.0 - p))

    @classmethod
    def oracle_uncertainty(
        cls, context: int, action: int
    ) -> tuple[float, float]:
        """
        Returns (p_success, bernoulli_variance p(1-p)) for paper action a.
        Used by MOPO oracle / debias penalty modes.
        """
        p = cls.success_probability(context, action)
        return float(p), float(p * (1.0 - p))

    def reward(self, state: int, action: int, success: bool) -> float:
        """
        논문 식 (4):
          성공: tanh(a / 28)  → 공격적 action일수록 보상이 큼 (throughput 장려)
          실패: −β · (k+1)   → 재전송이 늘어날수록 패널티 커짐
        """
        if success:
            # a/n_actions ∈ (0,1], tanh로 약간 압축
            return float(self.reward_scale * np.tanh(action / self.n_actions))
        # state=k (0-based), (k+1) = "몇 번째 시도였는지" 1-based 카운트
        return float(self.reward_scale * (-self.beta * (state + 1)))

    def next_state(self, state: int, success: bool) -> int:
        """
        HARQ 재전송 카운터 업데이트:
          - 성공 → 패킷 끝, 다음 패킷을 위해 k=0
          - 실패 & 아직 재시도 남음 → k+1
          - 실패 & 마지막 시도(k=n-1) → 포기하고 k=0 (드롭 후 새 패킷)
        """
        if success:
            return 0  # ACK → 새 패킷
        if state >= self.n_states - 1:
            return 0  # max retransmit 초과 → 드롭 후 리셋
        return state + 1  # NACK → 재전송 카운트 +1

    def paper_action(self, gym_action: int) -> int:
        """
        Gym Discrete 인덱스 → 논문 action 번호
          gym: 0, 1, ..., 27
          paper: 1, 2, ..., 28
        """
        if not self.action_space.contains(gym_action):
            raise ValueError(
                f"action {gym_action} outside Discrete({self.n_actions})"
            )
        return int(gym_action) + 1  # +1 매핑

    # ==================================================================
    # Gymnasium 표준 API: reset / step
    # ==================================================================

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, dict[str, Any]]:
        """
        에피소드 시작.
        반환: (observation, info)
          - options["state"]: 시작 k 강제 지정 가능
          - options["context"]: 시작 x 강제 지정 가능 (DP/디버깅용)
        """
        super().reset(seed=seed)  # Gymnasium 내부 시드 처리
        if seed is not None:
            # 우리 RNG도 같은 seed로 맞추기
            self._rng = np.random.default_rng(seed)

        options = options or {}  # None이면 빈 dict

        # 시작 상태 k (기본 0 = 새 패킷 첫 전송)
        self.state = int(options.get("state", 0))
        if not (0 <= self.state < self.n_states):
            raise ValueError(f"state must be in [0, {self.n_states})")

        # 시작 채널 context x
        if "context" in options:
            # 고정 context로 시작하고 싶을 때 (실험/재현)
            self.context = int(options["context"])
            if not (0 <= self.context < self.context_high):
                raise ValueError(
                    f"context must be in [0, {self.context_high})"
                )
        else:
            # 논문: x는 매 step마다 균등 샘플 → reset 시점에도 한 번 샘플
            self.context = int(self._rng.integers(0, self.context_high))

        self._step_count = 0  # 에피소드 스텝 카운터 리셋
        obs = self._get_obs()  # agent에게 줄 관측 벡터
        info = self._get_info(p_success=None, success=None, action=None)  # 디버그 정보
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        한 "전송 시도" 진행. Agent가 고른 MCS(action)로 패킷 한 번 보냄.

        반환 (Gymnasium 5-tuple):
          obs        : 다음 관측
          reward     : 이번 전송 보상
          terminated : 자연 종료 여부 (이 env는 항상 False)
          truncated  : 시간 제한으로 잘림 여부
          info       : 성공여부, p, 실제 MCS 번호 등
        """
        # 1) gym action(0..27) → paper MCS index(1..28)
        paper_a = self.paper_action(int(action))

        # 2) 현재 채널 x 와 고른 MCS a 로 성공 확률 계산
        p = self.success_probability(self.context, paper_a)

        # 3) 확률 p로 ACK(True) / NACK(False) 샘플
        #    U~Uniform(0,1), U < p 이면 성공
        success = bool(self._rng.random() < p)

        # 4) 성공/실패에 따른 즉시 보상
        reward = self.reward(self.state, paper_a, success)

        # 5) 재전송 카운터 k 업데이트
        self.state = self.next_state(self.state, success)

        # 6) 다음 전송을 위한 채널 품질을 새로 뽑음
        #    논문: context는 매 step i.i.d. (빠른 페이딩을 단순화)
        self.context = int(self._rng.integers(0, self.context_high))

        # 7) 에피소드 진행 카운트
        self._step_count += 1

        # 8) 종료 플래그
        #    terminated: 태스크 자체가 끝남 (goal 도달 등) → 여기선 없음
        #    truncated : 우리가 강제 컷 (max steps) → offline 데이터 길이 관리용
        terminated = False
        truncated = self._step_count >= self.max_episode_steps

        # 9) 다음 관측 + 로그용 info
        obs = self._get_obs()
        info = self._get_info(
            p_success=p, success=success, action=paper_a
        )
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """현재 내부 상태 → agent observation 벡터로 변환."""
        if self.include_context:
            # [k, x]: "몇 번째 시도인지" + "지금 채널 얼마나 좋은지"
            raw = np.array([self.state, self.context], dtype=np.float32)
            if self.normalize_obs:
                # 대략 [0,1] 스케일 (신경망 입력 안정화용, 기본은 꺼둠)
                raw[0] /= max(self.n_states - 1, 1)       # k 정규화
                raw[1] /= max(self.context_high - 1, 1)   # x 정규화
            return raw

        # 부분 관측: 채널 숨김 → agent는 재전송 횟수만 봄
        raw = np.array([float(self.state)], dtype=np.float32)
        if self.normalize_obs:
            raw[0] /= max(self.n_states - 1, 1)
        return raw

    def _get_info(
        self,
        p_success: Optional[float],
        success: Optional[bool],
        action: Optional[int],
    ) -> dict[str, Any]:
        """
        agent 학습에 안 쓰고, 로깅/디버깅용 부가정보.
        reset 직후에는 p/success/action 이 None일 수 있음.
        """
        return {
            "state": self.state,          # 업데이트 이후의 k
            "context": self.context,      # 업데이트 이후의 x (다음 step에 쓰일 값)
            "paper_action": action,       # 방금 사용한 MCS (1..28)
            "p_success": p_success,       # 방금 계산한 성공 확률
            "success": success,           # ACK/NACK 결과
            "step_count": self._step_count,
        }

    def render(self) -> None:
        # 시각화 미구현 (인터페이스만 맞춤)
        return None

    def close(self) -> None:
        # 리소스 정리 없음
        return None


def make_la_env(**kwargs: Any) -> LAEnv:
    """LAEnv 생성 헬퍼. 예: make_la_env(beta=0.3, seed=0)"""
    return LAEnv(**kwargs)
