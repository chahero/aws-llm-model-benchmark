"""
KoBEST WiC 벤치마크
Word in Context - 단어 문맥 의미 동일성 판단
"""
from typing import Dict, Any, List

from datasets import Dataset

from src.benchmarks.base_benchmark import BinaryClassificationBenchmark
from src.data.dataset_loader import dataset_loader


class WiCBenchmark(BinaryClassificationBenchmark):
    """KoBEST WiC 벤치마크"""

    def __init__(self):
        super().__init__(
            benchmark_name="kobest",
            task_name="wic",
            positive_label="True",
            negative_label="False",
        )

    def load_dataset(self, split: str = "test") -> Dataset:
        """데이터셋 로드"""
        return dataset_loader.load_kobest("wic", split)

    def format_prompt(self, example: Dict[str, Any]) -> str:
        """
        프롬프트 포맷

        Example 구조:
        - word: 대상 단어
        - context_1: 첫 번째 문맥
        - context_2: 두 번째 문맥
        - label: 0 (다름) 또는 1 (같음)
        """
        word = example["word"]
        context1 = example["context_1"]
        context2 = example["context_2"]

        prompt = f"""두 문장에서 "{word}"라는 단어가 같은 의미로 사용되었는지 판단하세요.

문장 1: {context1}
문장 2: {context2}

같은 의미이면 "True", 다른 의미이면 "False"로 답하세요.

답변:"""
        return prompt

    def get_reference(self, example: Dict[str, Any]) -> bool:
        """정답 추출"""
        return bool(example["label"])

    def get_system_prompt(self) -> str:
        return "당신은 단어의 문맥적 의미를 분석하는 AI입니다. True 또는 False로만 답하세요."
