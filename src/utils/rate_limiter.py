"""
API 호출 속도 제한
"""
import time
from collections import deque
from typing import Deque


class RateLimiter:
    """토큰 버킷 기반 속도 제한기"""

    def __init__(self, requests_per_second: float = 10.0):
        """
        Args:
            requests_per_second: 초당 허용 요청 수
        """
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0.0
        self.request_times: Deque[float] = deque(maxlen=int(requests_per_second * 2))

    def wait(self):
        """
        다음 요청 전 필요한 만큼 대기
        """
        current_time = time.time()

        # 이전 요청 이후 경과 시간
        elapsed = current_time - self.last_request_time

        # 최소 간격보다 적게 경과했으면 대기
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            time.sleep(sleep_time)

        self.last_request_time = time.time()
        self.request_times.append(self.last_request_time)

    def get_current_rate(self) -> float:
        """
        현재 요청 속도 계산 (최근 1초 기준)

        Returns:
            float: 초당 요청 수
        """
        if len(self.request_times) < 2:
            return 0.0

        current_time = time.time()
        # 최근 1초 이내의 요청만 카운트
        recent_requests = [t for t in self.request_times if current_time - t <= 1.0]
        return len(recent_requests)


class AdaptiveRateLimiter(RateLimiter):
    """적응형 속도 제한기 (에러 발생 시 자동 조절)"""

    def __init__(
        self,
        initial_rate: float = 10.0,
        min_rate: float = 1.0,
        max_rate: float = 50.0,
    ):
        super().__init__(initial_rate)
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.consecutive_errors = 0
        self.consecutive_success = 0

    def on_error(self):
        """에러 발생 시 속도 감소"""
        self.consecutive_errors += 1
        self.consecutive_success = 0

        # 속도 50% 감소
        new_rate = max(self.min_rate, self.requests_per_second * 0.5)
        self.requests_per_second = new_rate
        self.min_interval = 1.0 / new_rate

    def on_success(self):
        """성공 시 속도 점진적 증가"""
        self.consecutive_success += 1
        self.consecutive_errors = 0

        # 10번 연속 성공 시 속도 10% 증가
        if self.consecutive_success >= 10:
            new_rate = min(self.max_rate, self.requests_per_second * 1.1)
            self.requests_per_second = new_rate
            self.min_interval = 1.0 / new_rate
            self.consecutive_success = 0
