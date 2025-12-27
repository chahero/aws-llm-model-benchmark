"""
KLUE NLI 벤치마크
Natural Language Inference - 자연어 추론
"""
from typing import Dict, Any, List

from datasets import Dataset

from src.benchmarks.base_benchmark import MultipleChoiceBenchmark
from src.data.dataset_loader import dataset_loader


class NLIBenchmark(MultipleChoiceBenchmark):
    """KLUE NLI 벤치마크"""

    LABEL_MAP = {
        0: "entailment",
        1: "neutral",
        2: "contradiction",
    }

    LABEL_TO_CHOICE = {
        "entailment": "A",
        "neutral": "B",
        "contradiction": "C",
    }

    def __init__(self):
        super().__init__(
            benchmark_name="klue",
            task_name="nli",
            num_choices=3,
        )

    def load_dataset(self, split: str = "test") -> Dataset:
        """데이터셋 로드"""
        return dataset_loader.load_klue("nli", split)

    def format_prompt(self, example: Dict[str, Any]) -> str:
        """
        프롬프트 포맷

        Example 구조:
        - premise: 전제
        - hypothesis: 가설
        - label: 0(함의), 1(중립), 2(모순)
        """
        premise = example["premise"]
        hypothesis = example["hypothesis"]

        prompt = f"""두 문장 간의 관계를 판단하세요.

전제: {premise}
가설: {hypothesis}

관계:
A. 함의 (전제가 참이면 가설도 참)
B. 중립 (전제와 가설이 관련 없음)
C. 모순 (전제가 참이면 가설은 거짓)

정답 (A, B, C 중 하나):"""
        return prompt

    def get_reference(self, example: Dict[str, Any]) -> str:
        """정답 추출"""
        label = example["label"]
        label_str = self.LABEL_MAP[label]
        return self.LABEL_TO_CHOICE[label_str]

    def get_system_prompt(self) -> str:
        return "당신은 자연어 추론 전문가입니다. A(함의), B(중립), C(모순) 중 하나만 답하세요."
