"""
KoBEST BoolQ 벤치마크
Boolean Question Answering - 예/아니오 질문 답변
"""
from typing import Dict, Any, List

from datasets import Dataset

from src.benchmarks.base_benchmark import BinaryClassificationBenchmark
from src.data.dataset_loader import dataset_loader


class BoolQBenchmark(BinaryClassificationBenchmark):
    """KoBEST BoolQ 벤치마크"""

    def __init__(self):
        super().__init__(
            benchmark_name="kobest",
            task_name="boolq",
            positive_label="True",
            negative_label="False",
        )

    def load_dataset(self, split: str = "test") -> Dataset:
        """데이터셋 로드"""
        return dataset_loader.load_kobest("boolq", split)

    def format_prompt(self, example: Dict[str, Any]) -> str:
        """
        프롬프트 포맷

        Example 구조:
        - paragraph: 지문
        - question: 질문
        - label: 0 (False) 또는 1 (True)
        """
        paragraph = example["paragraph"]
        question = example["question"]

        prompt = f"""다음 지문을 읽고 질문에 대해 "True" 또는 "False"로만 답하세요.

지문:
{paragraph}

질문: {question}

답변:"""
        return prompt

    def get_reference(self, example: Dict[str, Any]) -> bool:
        """정답 추출"""
        return bool(example["label"])

    def get_system_prompt(self) -> str:
        return "당신은 질문에 대해 True 또는 False로만 답변하는 AI입니다. 다른 설명 없이 True 또는 False만 출력하세요."
