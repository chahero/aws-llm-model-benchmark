"""
KoBEST HellaSwag 벤치마크
문맥에 맞는 후속 문장 선택
"""
from typing import Dict, Any, List

from datasets import Dataset

from src.benchmarks.base_benchmark import MultipleChoiceBenchmark
from src.data.dataset_loader import dataset_loader


class HellaSwagBenchmark(MultipleChoiceBenchmark):
    """KoBEST HellaSwag 벤치마크"""

    def __init__(self):
        super().__init__(
            benchmark_name="kobest",
            task_name="hellaswag",
            num_choices=4,
        )

    def load_dataset(self, split: str = "test") -> Dataset:
        """데이터셋 로드"""
        return dataset_loader.load_kobest("hellaswag", split)

    def format_prompt(self, example: Dict[str, Any]) -> str:
        """
        프롬프트 포맷

        Example 구조:
        - context: 문맥
        - ending_1, ending_2, ending_3, ending_4: 선택지들
        - label: 0~3 (정답 인덱스)
        """
        context = example["context"]
        endings = [
            example["ending_1"],
            example["ending_2"],
            example["ending_3"],
            example["ending_4"],
        ]

        options = "\n".join(
            f"{label}. {ending}"
            for label, ending in zip(self.choice_labels, endings)
        )

        prompt = f"""다음 문맥에 이어질 가장 적절한 문장을 선택하세요.

문맥: {context}

선택지:
{options}

정답 (A, B, C, D 중 하나):"""
        return prompt

    def get_reference(self, example: Dict[str, Any]) -> str:
        """정답 추출"""
        label = example["label"]
        return self.choice_labels[label]

    def get_system_prompt(self) -> str:
        return "당신은 문맥에 맞는 후속 문장을 선택하는 AI입니다. A, B, C, D 중 하나만 답하세요."
