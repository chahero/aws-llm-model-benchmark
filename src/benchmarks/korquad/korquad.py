"""
KorQuAD 벤치마크
Korean Question Answering Dataset - 한국어 기계 독해
"""
import re
from typing import Dict, Any, List

from datasets import Dataset

from src.benchmarks.base_benchmark import BaseBenchmark
from src.data.dataset_loader import dataset_loader


class KorQuADBenchmark(BaseBenchmark):
    """KorQuAD 2.0 벤치마크 (기계 독해)"""

    def __init__(self):
        super().__init__(
            benchmark_name="korquad",
            task_name="v2",
        )

    def load_dataset(self, split: str = "test") -> Dataset:
        """데이터셋 로드 (KorQuAD는 test가 비공개라 validation 사용)"""
        # test 요청 시 validation으로 대체
        actual_split = "validation" if split == "test" else split
        return dataset_loader.load_korquad(actual_split)

    def format_prompt(self, example: Dict[str, Any]) -> str:
        """
        프롬프트 포맷

        Example 구조:
        - context: 지문
        - question: 질문
        - answer: {"text": [...], "answer_start": [...]}
        """
        context = example.get("context", "")
        question = example.get("question", "")

        # 컨텍스트가 너무 길면 자르기
        if len(context) > 2000:
            context = context[:2000] + "..."

        prompt = f"""다음 지문을 읽고 질문에 답하세요. 답은 지문에서 찾아 그대로 적어주세요.

지문:
{context}

질문: {question}

답변 (지문에서 발췌):"""
        return prompt

    def extract_answer(self, model_output: str) -> str:
        """모델 출력에서 답변 추출"""
        output = model_output.strip()

        # 줄바꿈이 있으면 첫 줄만
        if "\n" in output:
            output = output.split("\n")[0].strip()

        # 따옴표 제거
        output = output.strip('"\'')

        # "답변:" 등의 접두어 제거
        prefixes = ["답변:", "답:", "정답:", "Answer:"]
        for prefix in prefixes:
            if output.startswith(prefix):
                output = output[len(prefix):].strip()

        return output

    def get_reference(self, example: Dict[str, Any]) -> str:
        """정답 추출"""
        answer = example.get("answer", {})

        # KorQuAD 형식: {"text": ["답변"], "answer_start": [0]}
        if isinstance(answer, dict):
            texts = answer.get("text", [])
            if texts:
                return texts[0]

        # 단순 문자열인 경우
        if isinstance(answer, str):
            return answer

        return ""

    def _normalize_answer(self, text: str) -> str:
        """답변 정규화 (평가용)"""
        # 소문자 변환, 공백 정규화, 구두점 제거
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s가-힣]', '', text)
        return text.strip()

    def _compute_f1(self, prediction: str, reference: str) -> float:
        """F1 점수 계산"""
        pred_tokens = self._normalize_answer(prediction).split()
        ref_tokens = self._normalize_answer(reference).split()

        if not pred_tokens or not ref_tokens:
            return 0.0

        common = set(pred_tokens) & set(ref_tokens)
        if not common:
            return 0.0

        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def _compute_em(self, prediction: str, reference: str) -> float:
        """Exact Match 계산"""
        return float(self._normalize_answer(prediction) == self._normalize_answer(reference))

    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """평가 메트릭 계산"""
        total_f1 = 0.0
        total_em = 0.0

        for pred, ref in zip(predictions, references):
            total_f1 += self._compute_f1(pred, ref)
            total_em += self._compute_em(pred, ref)

        n = len(predictions)
        return {
            "f1": total_f1 / n if n > 0 else 0.0,
            "exact_match": total_em / n if n > 0 else 0.0,
        }

    def get_system_prompt(self) -> str:
        return "당신은 독해 전문가입니다. 지문에서 답을 찾아 정확히 인용하세요."
