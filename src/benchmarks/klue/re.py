"""
KLUE RE 벤치마크
Relation Extraction - 관계 추출
"""
from typing import Dict, Any, List

from datasets import Dataset

from src.benchmarks.base_benchmark import MultipleChoiceBenchmark
from src.data.dataset_loader import dataset_loader
from src.evaluators.metrics import calculate_f1


class REBenchmark(MultipleChoiceBenchmark):
    """KLUE RE 벤치마크"""

    # 관계 타입 (30개 클래스)
    RELATION_TYPES = [
        "no_relation",
        "org:dissolved",
        "org:founded",
        "org:place_of_headquarters",
        "org:alternate_names",
        "org:member_of",
        "org:members",
        "org:political/religious_affiliation",
        "org:product",
        "org:founded_by",
        "org:top_members/employees",
        "org:number_of_employees/members",
        "per:date_of_birth",
        "per:date_of_death",
        "per:place_of_birth",
        "per:place_of_death",
        "per:place_of_residence",
        "per:origin",
        "per:employee_of",
        "per:schools_attended",
        "per:alternate_names",
        "per:parents",
        "per:children",
        "per:siblings",
        "per:spouse",
        "per:other_family",
        "per:colleagues",
        "per:product",
        "per:religion",
        "per:title",
    ]

    def __init__(self):
        super().__init__(
            benchmark_name="klue",
            task_name="re",
            num_choices=len(self.RELATION_TYPES),
        )

    def load_dataset(self, split: str = "test") -> Dataset:
        """데이터셋 로드"""
        return dataset_loader.load_klue("re", split)

    def format_prompt(self, example: Dict[str, Any]) -> str:
        """
        프롬프트 포맷

        Example 구조:
        - sentence: 문장
        - subject_entity: {"word": ..., "start_idx": ..., "end_idx": ..., "type": ...}
        - object_entity: 같은 형식
        - label: 관계 레이블
        """
        sentence = example["sentence"]
        subject = example["subject_entity"]["word"]
        subject_type = example["subject_entity"]["type"]
        obj = example["object_entity"]["word"]
        obj_type = example["object_entity"]["type"]

        # 주요 관계만 선택지로 제시
        main_relations = [
            "no_relation (관계 없음)",
            "per:employee_of (소속)",
            "per:place_of_birth (출생지)",
            "per:spouse (배우자)",
            "org:founded_by (설립자)",
            "org:place_of_headquarters (본사 위치)",
        ]

        prompt = f"""다음 문장에서 두 개체 간의 관계를 파악하세요.

문장: {sentence}
주체: {subject} ({subject_type})
객체: {obj} ({obj_type})

가능한 관계:
{chr(10).join(main_relations)}

관계 (영어로 답하세요, 예: per:employee_of):"""
        return prompt

    def extract_answer(self, model_output: str) -> str:
        """관계 추출"""
        output = model_output.strip().lower()

        for relation in self.RELATION_TYPES:
            if relation.lower() in output:
                return relation

        return "no_relation"

    def get_reference(self, example: Dict[str, Any]) -> str:
        """정답 추출"""
        label = example["label"]
        # 정수형 인덱스를 문자열 레이블로 변환
        if isinstance(label, int):
            return self.RELATION_TYPES[label] if label < len(self.RELATION_TYPES) else "no_relation"
        return label

    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """평가 메트릭 계산"""
        accuracy = sum(p == r for p, r in zip(predictions, references)) / len(predictions)

        # no_relation 제외한 F1 계산
        filtered_preds = []
        filtered_refs = []
        for p, r in zip(predictions, references):
            if r != "no_relation":
                filtered_preds.append(p)
                filtered_refs.append(r)

        if filtered_refs:
            f1 = calculate_f1(filtered_preds, filtered_refs, average="micro")
        else:
            f1 = 0.0

        return {"accuracy": accuracy, "micro_f1": f1}

    def get_system_prompt(self) -> str:
        return "당신은 관계 추출 전문가입니다. 두 개체 간의 관계를 영어로 출력하세요."
