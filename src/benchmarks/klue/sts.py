"""
KLUE STS 벤치마크
Semantic Textual Similarity - 문장 유사도
"""
from typing import Dict, Any, List
import re

from datasets import Dataset

from src.benchmarks.base_benchmark import BaseBenchmark
from src.data.dataset_loader import dataset_loader
from src.evaluators.metrics import calculate_pearson_correlation


class STSBenchmark(BaseBenchmark):
    """KLUE STS 벤치마크"""

    def __init__(self):
        super().__init__(
            benchmark_name="klue",
            task_name="sts",
        )

    def load_dataset(self, split: str = "test") -> Dataset:
        """데이터셋 로드"""
        return dataset_loader.load_klue("sts", split)

    def format_prompt(self, example: Dict[str, Any]) -> str:
        """
        프롬프트 포맷

        Example 구조:
        - sentence1: 첫 번째 문장
        - sentence2: 두 번째 문장
        - labels: {"label": 0.0~5.0, ...}
        """
        sentence1 = example["sentence1"]
        sentence2 = example["sentence2"]

        prompt = f"""두 문장의 의미적 유사도를 0.0에서 5.0 사이의 점수로 평가하세요.
- 0.0: 완전히 다른 의미
- 5.0: 완전히 같은 의미

문장 1: {sentence1}
문장 2: {sentence2}

유사도 점수 (숫자만):"""
        return prompt

    def extract_answer(self, model_output: str) -> float:
        """
        모델 출력에서 점수 추출

        Args:
            model_output: 모델의 텍스트 출력

        Returns:
            float: 유사도 점수 (0.0~5.0)
        """
        output = model_output.strip()

        # 숫자 패턴 추출
        numbers = re.findall(r"\d+\.?\d*", output)
        if numbers:
            score = float(numbers[0])
            # 범위 제한
            return max(0.0, min(5.0, score))

        # 추출 실패 시 중간값 반환
        return 2.5

    def get_reference(self, example: Dict[str, Any]) -> float:
        """정답 추출"""
        labels = example.get("labels", {})
        return labels.get("label", 2.5)

    def evaluate(
        self,
        predictions: List[float],
        references: List[float],
    ) -> Dict[str, float]:
        """평가 메트릭 계산"""
        pearson = calculate_pearson_correlation(predictions, references)
        return {"pearson": pearson}

    def get_system_prompt(self) -> str:
        return "당신은 문장 유사도를 평가하는 AI입니다. 0.0에서 5.0 사이의 숫자만 출력하세요."
