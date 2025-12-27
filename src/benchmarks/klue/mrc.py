"""
KLUE MRC 벤치마크
Machine Reading Comprehension - 기계 독해
"""
from typing import Dict, Any, List

from datasets import Dataset

from src.benchmarks.base_benchmark import BaseBenchmark
from src.data.dataset_loader import dataset_loader
from src.evaluators.metrics import calculate_mrc_metrics


class MRCBenchmark(BaseBenchmark):
    """KLUE MRC 벤치마크"""

    def __init__(self):
        super().__init__(
            benchmark_name="klue",
            task_name="mrc",
        )

    def load_dataset(self, split: str = "test") -> Dataset:
        """데이터셋 로드"""
        return dataset_loader.load_klue("mrc", split)

    def format_prompt(self, example: Dict[str, Any]) -> str:
        """
        프롬프트 포맷

        Example 구조:
        - context: 지문
        - question: 질문
        - answers: {"answer_start": [...], "text": [...]}
        """
        context = example["context"]
        question = example["question"]

        prompt = f"""다음 지문을 읽고 질문에 답하세요. 지문에서 답을 그대로 추출하세요.

지문:
{context}

질문: {question}

답변:"""
        return prompt

    def extract_answer(self, model_output: str) -> str:
        """
        모델 출력에서 답변 추출

        Args:
            model_output: 모델의 텍스트 출력

        Returns:
            str: 추출된 답변
        """
        # 첫 줄 또는 첫 문장 추출
        answer = model_output.strip()

        # "답변:" 등의 접두사 제거
        prefixes = ["답변:", "답:", "정답:"]
        for prefix in prefixes:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()

        # 첫 줄만 사용
        if "\n" in answer:
            answer = answer.split("\n")[0].strip()

        return answer

    def get_reference(self, example: Dict[str, Any]) -> str:
        """정답 추출"""
        answers = example.get("answers", {})
        texts = answers.get("text", [])
        return texts[0] if texts else ""

    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """평가 메트릭 계산"""
        return calculate_mrc_metrics(predictions, references)

    def get_system_prompt(self) -> str:
        return "당신은 기계 독해 전문가입니다. 지문에서 질문에 대한 답을 정확히 추출하세요. 추가 설명 없이 답변만 출력하세요."
