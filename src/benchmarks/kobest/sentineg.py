"""
KoBEST SentiNeg 벤치마크
Sentiment Negation - 감정 부정 인식
"""
from typing import Dict, Any, List

from datasets import Dataset

from src.benchmarks.base_benchmark import MultipleChoiceBenchmark
from src.data.dataset_loader import dataset_loader


class SentiNegBenchmark(MultipleChoiceBenchmark):
    """KoBEST SentiNeg 벤치마크"""

    def __init__(self):
        super().__init__(
            benchmark_name="kobest",
            task_name="sentineg",
            num_choices=2,
        )

    def load_dataset(self, split: str = "test") -> Dataset:
        """데이터셋 로드"""
        return dataset_loader.load_kobest("sentineg", split)

    def format_prompt(self, example: Dict[str, Any]) -> str:
        """
        프롬프트 포맷

        Example 구조:
        - sentence: 문장
        - label: 0 (부정) 또는 1 (긍정)
        """
        sentence = example["sentence"]

        prompt = f"""다음 문장의 감정이 긍정인지 부정인지 판단하세요.

문장: {sentence}

긍정이면 "A", 부정이면 "B"로 답하세요.

답변:"""
        return prompt

    def get_reference(self, example: Dict[str, Any]) -> str:
        """정답 추출"""
        label = example["label"]
        return "A" if label == 1 else "B"  # 1=긍정=A, 0=부정=B

    def get_system_prompt(self) -> str:
        return "당신은 문장의 감정을 분석하는 AI입니다. A(긍정) 또는 B(부정) 중 하나만 답하세요."
