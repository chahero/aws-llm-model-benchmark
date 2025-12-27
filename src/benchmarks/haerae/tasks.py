"""
HAE-RAE 벤치마크 태스크들
한국어/문화 특화 평가
"""
from typing import Dict, Any, List

from datasets import Dataset

from src.benchmarks.base_benchmark import MultipleChoiceBenchmark
from src.data.dataset_loader import dataset_loader


class HaeRaeBaseBenchmark(MultipleChoiceBenchmark):
    """HAE-RAE 기본 벤치마크 클래스"""

    def __init__(self, task_name: str, num_choices: int = 4):
        super().__init__(
            benchmark_name="haerae",
            task_name=task_name,
            num_choices=num_choices,
        )

    def load_dataset(self, split: str = "test") -> Dataset:
        """데이터셋 로드"""
        return dataset_loader.load_haerae(self.task_name, split)

    def _get_choices_list(self, example: Dict[str, Any]) -> List[str]:
        """choices를 리스트로 변환 (다양한 데이터 구조 처리)"""
        # "choices" 또는 "options" 필드 시도
        # 빈 리스트일 경우에도 fallback 처리
        choices = example.get("choices") or example.get("options") or []

        # 이미 리스트인 경우
        if isinstance(choices, list):
            return choices

        # 딕셔너리인 경우 (예: {"A": "답1", "B": "답2", ...})
        if isinstance(choices, dict):
            return [choices.get(label, "") for label in self.choice_labels[:len(choices)]]

        return []

    def _get_answer_label(self, example: Dict[str, Any]) -> str:
        """정답 레이블 추출 (다양한 데이터 구조 처리)"""
        answer = example.get("answer", 0)
        choices = self._get_choices_list(example)

        # 이미 레이블 문자열인 경우 (예: "A", "B")
        if isinstance(answer, str):
            answer_upper = answer.strip().upper()
            if answer_upper in self.choice_labels:
                return answer_upper

            # "(A)", "(B)" 형식인 경우 - HAE-RAE 데이터셋
            import re
            bracket_match = re.match(r'^\s*\(([A-E])\)\s*$', answer_upper)
            if bracket_match:
                label = bracket_match.group(1)
                if label in self.choice_labels:
                    return label

            # 숫자 문자열인 경우 (예: "0", "1")
            try:
                idx = int(answer)
                if 0 <= idx < len(self.choice_labels):
                    return self.choice_labels[idx]
            except ValueError:
                pass

            # 정답이 텍스트인 경우 - choices에서 매칭되는 인덱스 찾기
            if choices:
                answer_normalized = answer.strip().lower()

                # 1단계: 정확히 일치
                for idx, choice in enumerate(choices):
                    if isinstance(choice, str):
                        choice_normalized = choice.strip().lower()
                        if answer_normalized == choice_normalized:
                            return self.choice_labels[idx] if idx < len(self.choice_labels) else "A"

                # 2단계: 공백/특수문자 제거 후 비교
                import re
                answer_clean = re.sub(r'[^\w가-힣]', '', answer_normalized)
                for idx, choice in enumerate(choices):
                    if isinstance(choice, str):
                        choice_clean = re.sub(r'[^\w가-힣]', '', choice.strip().lower())
                        if answer_clean == choice_clean:
                            return self.choice_labels[idx] if idx < len(self.choice_labels) else "A"

                # 3단계: 부분 매칭 (answer가 choice에 포함되거나 그 반대)
                for idx, choice in enumerate(choices):
                    if isinstance(choice, str):
                        choice_normalized = choice.strip().lower()
                        if answer_normalized in choice_normalized or choice_normalized in answer_normalized:
                            return self.choice_labels[idx] if idx < len(self.choice_labels) else "A"

            # 매칭 실패 - 기본값 반환
            # print(f"WARNING: Could not match answer '{answer[:50]}...' to {len(choices)} choices")
            return "A"

        # 정수 인덱스인 경우
        if isinstance(answer, int):
            if 0 <= answer < len(self.choice_labels):
                return self.choice_labels[answer]
            return "A"

        return "A"


class GeneralKnowledgeBenchmark(HaeRaeBaseBenchmark):
    """일반 상식 벤치마크"""

    def __init__(self):
        super().__init__("general_knowledge", num_choices=5)

    def format_prompt(self, example: Dict[str, Any]) -> str:
        query = example.get("query", "")
        choices = self._get_choices_list(example)

        options = "\n".join(
            f"{label}. {choice}"
            for label, choice in zip(self.choice_labels, choices)
        )

        prompt = f"""다음 질문에 답하세요.

질문: {query}

선택지:
{options}

정답 (A, B, C, D, E 중 하나):"""
        return prompt

    def get_reference(self, example: Dict[str, Any]) -> str:
        return self._get_answer_label(example)

    def get_system_prompt(self) -> str:
        return "당신은 한국 상식 전문가입니다. A, B, C, D, E 중 하나만 답하세요."


class HistoryBenchmark(HaeRaeBaseBenchmark):
    """한국사 벤치마크"""

    def __init__(self):
        super().__init__("history", num_choices=5)

    def format_prompt(self, example: Dict[str, Any]) -> str:
        query = example.get("query", "")
        choices = self._get_choices_list(example)

        options = "\n".join(
            f"{label}. {choice}"
            for label, choice in zip(self.choice_labels, choices)
        )

        prompt = f"""다음 한국사 관련 질문에 답하세요.

질문: {query}

선택지:
{options}

정답 (A, B, C, D, E 중 하나):"""
        return prompt

    def get_reference(self, example: Dict[str, Any]) -> str:
        return self._get_answer_label(example)

    def get_system_prompt(self) -> str:
        return "당신은 한국사 전문가입니다. A, B, C, D, E 중 하나만 답하세요."


class ReadingComprehensionBenchmark(HaeRaeBaseBenchmark):
    """독해 벤치마크"""

    def __init__(self):
        super().__init__("reading_comprehension", num_choices=5)

    def format_prompt(self, example: Dict[str, Any]) -> str:
        paragraph = example.get("paragraph", "")
        query = example.get("query", "")
        choices = self._get_choices_list(example)

        options = "\n".join(
            f"{label}. {choice}"
            for label, choice in zip(self.choice_labels, choices)
        )

        prompt = f"""다음 지문을 읽고 질문에 답하세요.

지문:
{paragraph}

질문: {query}

선택지:
{options}

정답 (A, B, C, D, E 중 하나):"""
        return prompt

    def get_reference(self, example: Dict[str, Any]) -> str:
        return self._get_answer_label(example)

    def get_system_prompt(self) -> str:
        return "당신은 독해 전문가입니다. 지문을 읽고 A, B, C, D, E 중 하나만 답하세요."


class StandardNomenclatureBenchmark(HaeRaeBaseBenchmark):
    """표준어/맞춤법 벤치마크"""

    def __init__(self):
        super().__init__("standard_nomenclature", num_choices=5)

    def format_prompt(self, example: Dict[str, Any]) -> str:
        query = example.get("query", "")
        choices = self._get_choices_list(example)

        options = "\n".join(
            f"{label}. {choice}"
            for label, choice in zip(self.choice_labels, choices)
        )

        prompt = f"""다음 중 올바른 표기/표현을 선택하세요.

질문: {query}

선택지:
{options}

정답 (A, B, C, D, E 중 하나):"""
        return prompt

    def get_reference(self, example: Dict[str, Any]) -> str:
        return self._get_answer_label(example)

    def get_system_prompt(self) -> str:
        return "당신은 한국어 맞춤법 전문가입니다. A, B, C, D, E 중 하나만 답하세요."


class LoanWordBenchmark(HaeRaeBaseBenchmark):
    """외래어 표기 벤치마크"""

    def __init__(self):
        super().__init__("loan_words", num_choices=5)

    def format_prompt(self, example: Dict[str, Any]) -> str:
        query = example.get("query", "")
        choices = self._get_choices_list(example)

        options = "\n".join(
            f"{label}. {choice}"
            for label, choice in zip(self.choice_labels, choices)
        )

        prompt = f"""다음 외래어의 올바른 표기를 선택하세요.

질문: {query}

선택지:
{options}

정답 (A, B, C, D, E 중 하나):"""
        return prompt

    def get_reference(self, example: Dict[str, Any]) -> str:
        return self._get_answer_label(example)

    def get_system_prompt(self) -> str:
        return "당신은 외래어 표기 전문가입니다. A, B, C, D, E 중 하나만 답하세요."


class RareWordBenchmark(HaeRaeBaseBenchmark):
    """희귀 단어 벤치마크"""

    def __init__(self):
        super().__init__("rare_words", num_choices=5)

    def format_prompt(self, example: Dict[str, Any]) -> str:
        query = example.get("query", "")
        choices = self._get_choices_list(example)

        options = "\n".join(
            f"{label}. {choice}"
            for label, choice in zip(self.choice_labels, choices)
        )

        prompt = f"""다음 단어의 의미를 선택하세요.

단어: {query}

선택지:
{options}

정답 (A, B, C, D, E 중 하나):"""
        return prompt

    def get_reference(self, example: Dict[str, Any]) -> str:
        return self._get_answer_label(example)

    def get_system_prompt(self) -> str:
        return "당신은 한국어 어휘 전문가입니다. A, B, C, D, E 중 하나만 답하세요."


def get_all_haerae_benchmarks():
    """모든 HAE-RAE 벤치마크 반환"""
    return [
        GeneralKnowledgeBenchmark(),
        HistoryBenchmark(),
        ReadingComprehensionBenchmark(),
        StandardNomenclatureBenchmark(),
        LoanWordBenchmark(),
        RareWordBenchmark(),
    ]
