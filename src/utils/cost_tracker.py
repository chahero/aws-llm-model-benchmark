"""
비용 추적 유틸리티
"""
from dataclasses import dataclass, field
from typing import Dict, List
from datetime import datetime

from config.settings import MODEL_COSTS


@dataclass
class UsageRecord:
    """단일 사용 기록"""
    model_name: str
    input_tokens: int
    output_tokens: int
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def cost(self) -> float:
        """이 기록의 비용 계산"""
        if self.model_name not in MODEL_COSTS:
            return 0.0

        costs = MODEL_COSTS[self.model_name]
        input_cost = (self.input_tokens / 1000) * costs["input"]
        output_cost = (self.output_tokens / 1000) * costs["output"]
        return input_cost + output_cost


class CostTracker:
    """비용 추적기"""

    def __init__(self):
        self.records: List[UsageRecord] = []
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0

    def add_usage(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
    ):
        """
        사용 기록 추가

        Args:
            model_name: 모델 이름
            input_tokens: 입력 토큰 수
            output_tokens: 출력 토큰 수
        """
        record = UsageRecord(
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        self.records.append(record)
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    def get_total_cost(self) -> float:
        """전체 비용 계산"""
        return sum(record.cost for record in self.records)

    def get_cost_by_model(self) -> Dict[str, float]:
        """모델별 비용 계산"""
        costs: Dict[str, float] = {}
        for record in self.records:
            if record.model_name not in costs:
                costs[record.model_name] = 0.0
            costs[record.model_name] += record.cost
        return costs

    def get_summary(self) -> Dict:
        """사용량 요약 반환"""
        return {
            "total_requests": len(self.records),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": self.get_total_cost(),
            "cost_by_model": self.get_cost_by_model(),
        }

    def reset(self):
        """기록 초기화"""
        self.records = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
