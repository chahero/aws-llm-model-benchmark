"""
벤치마크 추상 베이스 클래스
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Generator, Optional
from datetime import datetime

from datasets import Dataset


@dataclass
class BenchmarkResult:
    """벤치마크 결과 데이터 클래스"""
    benchmark_name: str
    task_name: str
    model_name: str
    predictions: List[Any]
    references: List[Any]
    metrics: Dict[str, float]
    raw_outputs: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "benchmark_name": self.benchmark_name,
            "task_name": self.task_name,
            "model_name": self.model_name,
            "metrics": self.metrics,
            "metadata": self.metadata,
            "num_examples": len(self.predictions),
            "raw_outputs": self.raw_outputs,
        }


class BaseBenchmark(ABC):
    """벤치마크 추상 베이스 클래스"""

    def __init__(self, benchmark_name: str, task_name: str):
        """
        Args:
            benchmark_name: 벤치마크 이름 (klue, kobest, haerae)
            task_name: 태스크 이름
        """
        self.benchmark_name = benchmark_name
        self.task_name = task_name
        self._dataset: Optional[Dataset] = None

    @abstractmethod
    def load_dataset(self, split: str = "test") -> Dataset:
        """
        데이터셋 로드

        Args:
            split: 데이터 분할 (train, validation, test)

        Returns:
            Dataset: HuggingFace Dataset 객체
        """
        pass

    @abstractmethod
    def format_prompt(self, example: Dict[str, Any]) -> str:
        """
        예제를 프롬프트로 포맷

        Args:
            example: 데이터셋의 단일 예제

        Returns:
            str: 모델 입력 프롬프트
        """
        pass

    @abstractmethod
    def extract_answer(self, model_output: str) -> Any:
        """
        모델 출력에서 답변 추출

        Args:
            model_output: 모델의 텍스트 출력

        Returns:
            Any: 추출된 답변
        """
        pass

    @abstractmethod
    def get_reference(self, example: Dict[str, Any]) -> Any:
        """
        예제에서 정답 추출

        Args:
            example: 데이터셋의 단일 예제

        Returns:
            Any: 정답 레이블
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        predictions: List[Any],
        references: List[Any],
    ) -> Dict[str, float]:
        """
        예측 결과 평가

        Args:
            predictions: 예측값 리스트
            references: 정답 리스트

        Returns:
            Dict[str, float]: 평가 메트릭
        """
        pass

    def get_system_prompt(self) -> str:
        """
        시스템 프롬프트 반환 (오버라이드 가능)

        Returns:
            str: 시스템 프롬프트
        """
        return "당신은 한국어 자연어 처리 전문가입니다. 주어진 질문에 정확하게 답변해주세요."

    def iterate_examples(
        self,
        split: str = "test",
        limit: Optional[int] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        데이터셋 이터레이터

        Args:
            split: 데이터 분할
            limit: 최대 예제 수 (None이면 전체)

        Yields:
            Dict[str, Any]: 데이터셋 예제
        """
        dataset = self.load_dataset(split)

        for i, example in enumerate(dataset):
            if limit is not None and i >= limit:
                break
            yield example

    def get_num_examples(self, split: str = "test") -> int:
        """데이터셋 크기 반환"""
        dataset = self.load_dataset(split)
        return len(dataset)

    def get_task_info(self) -> Dict[str, Any]:
        """태스크 정보 반환"""
        return {
            "benchmark_name": self.benchmark_name,
            "task_name": self.task_name,
            "system_prompt": self.get_system_prompt(),
        }


class MultipleChoiceBenchmark(BaseBenchmark):
    """객관식 문제 벤치마크 베이스 클래스"""

    def __init__(
        self,
        benchmark_name: str,
        task_name: str,
        num_choices: int = 4,
    ):
        super().__init__(benchmark_name, task_name)
        self.num_choices = num_choices
        self.choice_labels = ["A", "B", "C", "D", "E"][:num_choices]

    def extract_answer(self, model_output: str) -> str:
        """
        모델 출력에서 선택지 추출

        Args:
            model_output: 모델의 텍스트 출력

        Returns:
            str: 선택된 라벨 (A, B, C, D 등)
        """
        import re

        output = model_output.strip()
        output_upper = output.upper()

        # 직접 라벨인 경우 (출력이 단순히 "A", "B" 등)
        if output_upper in self.choice_labels:
            return output_upper

        # 패턴 기반 추출 (우선순위 순)
        labels_pattern = "|".join(self.choice_labels)

        # 0. 첫 줄이 단일 라벨인 경우 (예: "A\n\n설명...")
        first_line = output_upper.split('\n')[0].strip()
        if first_line in self.choice_labels:
            return first_line

        # 1. "정답: (A)" 또는 "정답: A" 패턴
        match = re.search(rf'정답[:\s]*[\(\[]?({labels_pattern})[\)\]]?', output_upper)
        if match:
            return match.group(1)

        # 2. "**(A)**" 또는 "**A**" 패턴 (마크다운 볼드)
        match = re.search(rf'\*\*[\(\[]?({labels_pattern})[\)\]]?\*\*', output_upper)
        if match:
            return match.group(1)

        # 3. "(A)" 또는 "[A]" 패턴 (첫 번째 출현)
        match = re.search(rf'[\(\[]({labels_pattern})[\)\]]', output_upper)
        if match:
            return match.group(1)

        # 4. "A." 또는 "A)" 또는 "A:" 패턴 (문장 시작 부분)
        match = re.search(rf'^[\s]*({labels_pattern})[.\):\s]', output_upper, re.MULTILINE)
        if match:
            return match.group(1)

        # 5. 단독 레이블 (공백으로 구분된 경우)
        match = re.search(rf'\b({labels_pattern})\b', output_upper)
        if match:
            return match.group(1)

        # 추출 실패 시 첫 번째 선택지 반환
        return self.choice_labels[0]

    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """정확도 계산"""
        if not predictions or not references:
            return {"accuracy": 0.0}

        correct = sum(p == r for p, r in zip(predictions, references))
        accuracy = correct / len(predictions)

        return {"accuracy": accuracy}


class BinaryClassificationBenchmark(BaseBenchmark):
    """이진 분류 벤치마크 베이스 클래스"""

    def __init__(
        self,
        benchmark_name: str,
        task_name: str,
        positive_label: str = "True",
        negative_label: str = "False",
    ):
        super().__init__(benchmark_name, task_name)
        self.positive_label = positive_label
        self.negative_label = negative_label

    def extract_answer(self, model_output: str) -> bool:
        """
        모델 출력에서 True/False 추출

        Args:
            model_output: 모델의 텍스트 출력

        Returns:
            bool: 예측 결과
        """
        import re

        output = model_output.strip().lower()

        # 1. 첫 단어 확인 (가장 신뢰할 수 있는 방법)
        first_word = output.split()[0] if output.split() else ""
        # 특수문자 제거
        first_word_clean = re.sub(r'[^a-z가-힣]', '', first_word)

        if first_word_clean in ["true", "yes", "예", "맞습니다", "맞음", "참"]:
            return True
        if first_word_clean in ["false", "no", "아니오", "아니요", "틀립니다", "틀림", "거짓"]:
            return False

        # 2. 첫 줄에서 True/False 확인
        first_line = output.split('\n')[0].strip().lower()
        if first_line in ["true", "false", "yes", "no"]:
            return first_line in ["true", "yes"]

        # 3. 단어 경계를 사용한 패턴 매칭 (부분 문자열 매칭 방지)
        # True 패턴
        if re.search(r'\btrue\b', output):
            return True
        if re.search(r'\bfalse\b', output):
            return False

        # 4. 한국어 패턴 (문장 시작 부분에서만)
        if re.match(r'^(예|맞습니다|맞음|참)', output):
            return True
        if re.match(r'^(아니오|아니요|틀립니다|틀림|거짓)', output):
            return False

        # 기본값: False
        return False

    def evaluate(
        self,
        predictions: List[bool],
        references: List[bool],
    ) -> Dict[str, float]:
        """정확도 계산"""
        if not predictions or not references:
            return {"accuracy": 0.0}

        correct = sum(p == r for p, r in zip(predictions, references))
        accuracy = correct / len(predictions)

        return {"accuracy": accuracy}
