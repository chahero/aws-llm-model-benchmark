"""
KLUE YNAT 벤치마크
Topic Classification - 뉴스 주제 분류
"""
from typing import Dict, Any, List

from datasets import Dataset

from src.benchmarks.base_benchmark import MultipleChoiceBenchmark
from src.data.dataset_loader import dataset_loader
from src.evaluators.metrics import calculate_f1


class YNATBenchmark(MultipleChoiceBenchmark):
    """KLUE YNAT 벤치마크"""

    LABEL_MAP = {
        0: "IT/과학",
        1: "경제",
        2: "사회",
        3: "생활/문화",
        4: "세계",
        5: "스포츠",
        6: "정치",
    }

    def __init__(self):
        super().__init__(
            benchmark_name="klue",
            task_name="ynat",
            num_choices=7,
        )
        self.choice_labels = ["A", "B", "C", "D", "E", "F", "G"]

    def load_dataset(self, split: str = "test") -> Dataset:
        """데이터셋 로드"""
        return dataset_loader.load_klue("ynat", split)

    def format_prompt(self, example: Dict[str, Any]) -> str:
        """
        프롬프트 포맷

        Example 구조:
        - title: 뉴스 제목
        - label: 0~6 (주제 카테고리)
        """
        title = example["title"]

        prompt = f"""다음 뉴스 제목의 주제를 분류하세요.

뉴스 제목: {title}

주제:
A. IT/과학
B. 경제
C. 사회
D. 생활/문화
E. 세계
F. 스포츠
G. 정치

정답 (A~G 중 하나):"""
        return prompt

    def get_reference(self, example: Dict[str, Any]) -> str:
        """정답 추출"""
        label = example["label"]
        return self.choice_labels[label]

    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """평가 메트릭 계산"""
        accuracy = sum(p == r for p, r in zip(predictions, references)) / len(predictions)
        f1 = calculate_f1(predictions, references, average="macro")
        return {"accuracy": accuracy, "macro_f1": f1}

    def get_system_prompt(self) -> str:
        return "당신은 뉴스 분류 전문가입니다. A~G 중 하나만 답하세요."
