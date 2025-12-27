"""
KoBEST COPA 벤치마크
Choice of Plausible Alternatives - 인과관계 추론
"""
from typing import Dict, Any, List

from datasets import Dataset

from src.benchmarks.base_benchmark import MultipleChoiceBenchmark
from src.data.dataset_loader import dataset_loader


class COPABenchmark(MultipleChoiceBenchmark):
    """KoBEST COPA 벤치마크"""

    def __init__(self):
        super().__init__(
            benchmark_name="kobest",
            task_name="copa",
            num_choices=2,
        )

    def load_dataset(self, split: str = "test") -> Dataset:
        """데이터셋 로드"""
        return dataset_loader.load_kobest("copa", split)

    def format_prompt(self, example: Dict[str, Any]) -> str:
        """
        프롬프트 포맷

        Example 구조:
        - premise: 전제
        - question: "원인" 또는 "결과"
        - alternative_1: 선택지 1
        - alternative_2: 선택지 2
        - label: 0 또는 1
        """
        premise = example["premise"]
        question = example["question"]
        alt1 = example["alternative_1"]
        alt2 = example["alternative_2"]

        question_type = "원인" if question == "원인" else "결과"

        prompt = f"""다음 문장의 {question_type}으로 가장 적절한 것을 A 또는 B 중에서 고르세요.

문장: {premise}

A. {alt1}
B. {alt2}

정답 (A 또는 B):"""
        return prompt

    def get_reference(self, example: Dict[str, Any]) -> str:
        """정답 추출"""
        label = example["label"]
        return "A" if label == 0 else "B"

    def get_system_prompt(self) -> str:
        return "당신은 인과관계를 추론하는 AI입니다. A 또는 B 중 하나만 답하세요."
