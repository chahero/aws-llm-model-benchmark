"""
KMMLU 벤치마크
Korean Massive Multitask Language Understanding - 한국어 다분야 지식 평가
"""
from typing import Dict, Any, List

from datasets import Dataset, load_dataset

from src.benchmarks.base_benchmark import MultipleChoiceBenchmark
from src.utils.logger import get_logger

logger = get_logger(__name__)

# KMMLU 카테고리 (실제 HuggingFace 데이터셋 기준)
KMMLU_CATEGORIES = [
    "Accounting",
    "Agricultural-Sciences",
    "Aviation-Engineering-and-Maintenance",
    "Biology",
    "Chemical-Engineering",
    "Chemistry",
    "Civil-Engineering",
    "Computer-Science",
    "Construction",
    "Criminal-Law",
    "Ecology",
    "Economics",
    "Education",
    "Electrical-Engineering",
    "Electronics-Engineering",
    "Energy-Management",
    "Environmental-Science",
    "Fashion",
    "Food-Processing",
    "Gas-Technology-and-Engineering",
    "Geomatics",
    "Health",
    "Industrial-Engineer",
    "Information-Technology",
    "Interior-Architecture-and-Design",
    "Law",
    "Machine-Design-and-Manufacturing",
    "Management",
    "Maritime-Engineering",
    "Marketing",
    "Materials-Engineering",
    "Mechanical-Engineering",
    "Nondestructive-Testing",
    "Patent",
    "Political-Science-and-Sociology",
    "Psychology",
    "Public-Safety",
    "Railway-and-Automotive-Engineering",
    "Real-Estate",
    "Refrigerating-Machinery",
    "Social-Welfare",
    "Taxation",
    "Telecommunications-and-Wireless-Technology",
    "Korean-History",
    "Math",
]


class KMMLUBenchmark(MultipleChoiceBenchmark):
    """KMMLU 벤치마크 (다분야 지식 평가)"""

    def __init__(self, categories: List[str] = None):
        super().__init__(
            benchmark_name="kmmlu",
            task_name="all",
            num_choices=4,
        )
        self.categories = categories or KMMLU_CATEGORIES

    def load_dataset(self, split: str = "test") -> Dataset:
        """모든 카테고리의 데이터셋 로드 및 병합"""
        all_examples = []

        for category in self.categories:
            try:
                ds = load_dataset("HAERAE-HUB/KMMLU", category, split=split)
                for example in ds:
                    example["_category"] = category
                    all_examples.append(example)
            except Exception as e:
                logger.warning(f"Failed to load KMMLU category {category}: {e}")
                continue

        logger.info(f"Loaded {len(all_examples)} examples from {len(self.categories)} categories")

        # Dataset 객체로 변환
        if all_examples:
            from datasets import Dataset as HFDataset
            return HFDataset.from_list(all_examples)
        return None

    def format_prompt(self, example: Dict[str, Any]) -> str:
        """
        프롬프트 포맷

        Example 구조:
        - question: 질문
        - A, B, C, D: 선택지
        - answer: 정답 (1, 2, 3, 4 또는 A, B, C, D)
        - Category: 카테고리
        """
        question = example.get("question", "")
        category = example.get("_category", example.get("Category", ""))

        # 선택지 추출
        choices = []
        for label in ["A", "B", "C", "D"]:
            choice = example.get(label, "")
            if choice:
                choices.append(f"{label}. {choice}")

        options = "\n".join(choices)

        # 카테고리 한글화 (일부)
        category_kr = self._translate_category(category)

        prompt = f"""[{category_kr}] 다음 질문에 답하세요.

질문: {question}

선택지:
{options}

정답 (A, B, C, D 중 하나):"""
        return prompt

    def _translate_category(self, category: str) -> str:
        """카테고리 한글 번역"""
        translations = {
            "Accounting": "회계학",
            "Biology": "생물학",
            "Chemistry": "화학",
            "Computer-Science": "컴퓨터과학",
            "Economics": "경제학",
            "Education": "교육학",
            "Law": "법학",
            "Math": "수학",
            "Psychology": "심리학",
            "Korean-History": "한국사",
            "Geography": "지리학",
            "Health": "보건",
            "English": "영어",
            "Korean-Language-and-Literature": "국어국문학",
            "Political-Science-and-Sociology": "정치사회학",
        }
        return translations.get(category, category)

    def get_reference(self, example: Dict[str, Any]) -> str:
        """정답 추출"""
        answer = example.get("answer", "")

        # 정수형 (1, 2, 3, 4) -> A, B, C, D
        if isinstance(answer, int):
            if 1 <= answer <= 4:
                return self.choice_labels[answer - 1]
            if 0 <= answer <= 3:
                return self.choice_labels[answer]

        # 문자열
        if isinstance(answer, str):
            answer = answer.strip().upper()
            if answer in self.choice_labels:
                return answer
            # 숫자 문자열
            try:
                idx = int(answer)
                if 1 <= idx <= 4:
                    return self.choice_labels[idx - 1]
            except ValueError:
                pass

        return "A"

    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """평가 메트릭 계산"""
        correct = sum(p == r for p, r in zip(predictions, references))
        accuracy = correct / len(predictions) if predictions else 0.0

        return {"accuracy": accuracy}

    def get_system_prompt(self) -> str:
        return "당신은 다양한 분야의 전문가입니다. A, B, C, D 중 하나만 답하세요."
