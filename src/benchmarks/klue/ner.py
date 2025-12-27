"""
KLUE NER 벤치마크
Named Entity Recognition - 개체명 인식
"""
from typing import Dict, Any, List
import re

from datasets import Dataset

from src.benchmarks.base_benchmark import BaseBenchmark
from src.data.dataset_loader import dataset_loader


class NERBenchmark(BaseBenchmark):
    """KLUE NER 벤치마크"""

    ENTITY_TYPES = ["PS", "LC", "OG", "DT", "TI", "QT"]

    # KLUE NER 태그 ID -> 문자열 매핑 (올바른 매핑)
    # 참고: https://huggingface.co/datasets/klue/viewer/ner
    TAG_MAP = {
        0: "B-DT", 1: "I-DT",      # Date/Time
        2: "B-LC", 3: "I-LC",      # Location
        4: "B-OG", 5: "I-OG",      # Organization
        6: "B-PS", 7: "I-PS",      # Person
        8: "B-QT", 9: "I-QT",      # Quantity
        10: "B-TI", 11: "I-TI",    # Time (별도)
        12: "O",                    # Outside
    }

    def __init__(self):
        super().__init__(
            benchmark_name="klue",
            task_name="ner",
        )

    def load_dataset(self, split: str = "test") -> Dataset:
        """데이터셋 로드"""
        return dataset_loader.load_klue("ner", split)

    def format_prompt(self, example: Dict[str, Any]) -> str:
        """
        프롬프트 포맷

        Example 구조:
        - tokens: 토큰 리스트
        - ner_tags: NER 태그 리스트
        """
        tokens = example.get("tokens", [])
        sentence = " ".join(tokens)

        prompt = f"""다음 문장에서 개체명을 추출하세요. 형식: [개체명:타입]
타입: PS(인물), LC(장소), OG(기관), DT(날짜), TI(시간), QT(수량)

문장: {sentence}

개체명 (없으면 "없음"):"""
        return prompt

    def extract_answer(self, model_output: str) -> List[Dict[str, str]]:
        """
        모델 출력에서 개체명 추출

        Returns:
            List[Dict]: [{"entity": "...", "type": "..."}]
        """
        output = model_output.strip()

        if "없음" in output or not output:
            return []

        entities = []

        # 패턴 1: [개체명:타입] 값 형식 (예: [개체명:PS] 홍길동)
        pattern1 = r"\[개체명:([A-Z]{2})\]\s*([^\n\[\]]+)"
        matches1 = re.findall(pattern1, output)
        for entity_type, entity in matches1:
            entity = entity.strip().rstrip(',').strip()
            if entity_type in self.ENTITY_TYPES and entity:
                entities.append({"entity": entity, "type": entity_type})

        # 패턴 2: [값:타입] 형식 (예: [홍길동:PS])
        if not entities:
            pattern2 = r"\[([^:\[\]]+):([A-Z]{2})\]"
            matches2 = re.findall(pattern2, output)
            for entity, entity_type in matches2:
                entity = entity.strip()
                if entity_type in self.ENTITY_TYPES and entity and entity != "개체명":
                    entities.append({"entity": entity, "type": entity_type})

        # 패턴 3: - [타입] 값 형식 (예: - [PS] 홍길동)
        if not entities:
            pattern3 = r"-?\s*\[([A-Z]{2})\]\s*([^\n\[\]]+)"
            matches3 = re.findall(pattern3, output)
            for entity_type, entity in matches3:
                entity = entity.strip().rstrip(',').strip()
                if entity_type in self.ENTITY_TYPES and entity:
                    entities.append({"entity": entity, "type": entity_type})

        return entities

    def get_reference(self, example: Dict[str, Any]) -> List[Dict[str, str]]:
        """정답 추출"""
        tokens = example.get("tokens", [])
        ner_tags = example.get("ner_tags", [])

        entities = []
        current_entity = []
        current_type = None

        for token, tag_id in zip(tokens, ner_tags):
            # 정수형 태그를 문자열로 변환
            tag = self.TAG_MAP.get(tag_id, "O")

            if tag.startswith("B-"):
                if current_entity:
                    # 문자 단위 토큰이므로 공백 없이 합침
                    entities.append({
                        "entity": "".join(current_entity),
                        "type": current_type
                    })
                current_entity = [token]
                current_type = tag[2:]
            elif tag.startswith("I-") and current_entity and tag[2:] == current_type:
                # I-태그는 현재 엔티티 타입과 같을 때만 추가
                current_entity.append(token)
            else:
                if current_entity:
                    entities.append({
                        "entity": "".join(current_entity),
                        "type": current_type
                    })
                current_entity = []
                current_type = None

        if current_entity:
            entities.append({
                "entity": "".join(current_entity),
                "type": current_type
            })

        return entities

    def _normalize_entity(self, entity: str) -> str:
        """엔티티 문자열 정규화 (공백 제거)"""
        return entity.replace(" ", "").strip()

    def evaluate(
        self,
        predictions: List[List[Dict]],
        references: List[List[Dict]],
    ) -> Dict[str, float]:
        """F1 점수 계산 (간소화 버전)"""
        total_pred = 0
        total_ref = 0
        total_correct = 0

        for pred_entities, ref_entities in zip(predictions, references):
            # 공백 정규화하여 비교 (모델 출력은 공백 포함, 참조는 공백 미포함)
            pred_set = {(self._normalize_entity(e["entity"]), e["type"]) for e in pred_entities}
            ref_set = {(self._normalize_entity(e["entity"]), e["type"]) for e in ref_entities}

            total_pred += len(pred_set)
            total_ref += len(ref_set)
            total_correct += len(pred_set & ref_set)

        precision = total_correct / total_pred if total_pred > 0 else 0
        recall = total_correct / total_ref if total_ref > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {"precision": precision, "recall": recall, "f1": f1}

    def get_system_prompt(self) -> str:
        return "당신은 개체명 인식 전문가입니다. [개체명:타입] 형식으로 출력하세요."
