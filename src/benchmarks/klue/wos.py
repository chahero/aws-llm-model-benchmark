"""
KLUE WOS 벤치마크
Dialogue State Tracking - 대화 상태 추적
"""
from typing import Dict, Any, List, Set
import json
import re

from datasets import Dataset

from src.benchmarks.base_benchmark import BaseBenchmark
from src.data.dataset_loader import dataset_loader


class WOSBenchmark(BaseBenchmark):
    """KLUE WOS 벤치마크"""

    DOMAINS = ["관광", "숙소", "식당", "지하철", "택시"]

    # 각 도메인별 슬롯 정의
    SLOTS = {
        "관광": ["관광-경치 좋은", "관광-교육적", "관광-도보 가능", "관광-문화 예술",
                "관광-역사적", "관광-이름", "관광-종류", "관광-주차 가능", "관광-지역"],
        "숙소": ["숙소-가격대", "숙소-도보 가능", "숙소-수영장 유무", "숙소-스파 유무",
                "숙소-식당 유무", "숙소-예약 기간", "숙소-예약 명수", "숙소-예약 요일",
                "숙소-이름", "숙소-인터넷 가능", "숙소-조식 가능", "숙소-종류",
                "숙소-주차 가능", "숙소-지역", "숙소-헬스장 유무"],
        "식당": ["식당-가격대", "식당-도보 가능", "식당-야외석 유무", "식당-예약 명수",
                "식당-예약 시간", "식당-예약 요일", "식당-이름", "식당-인터넷 가능",
                "식당-주류 판매", "식당-주차 가능", "식당-종류", "식당-지역", "식당-흡연 가능"],
        "지하철": ["지하철-도착역", "지하철-출발역"],
        "택시": ["택시-도착지", "택시-종류", "택시-출발지"]
    }

    def __init__(self):
        super().__init__(
            benchmark_name="klue",
            task_name="wos",
        )

    def load_dataset(self, split: str = "test") -> Dataset:
        """데이터셋 로드"""
        return dataset_loader.load_klue("wos", split)

    def format_prompt(self, example: Dict[str, Any]) -> str:
        """프롬프트 포맷"""
        dialogue = example.get("dialogue", [])

        # 전체 대화 기록 구성
        dialogue_text = ""
        for turn in dialogue:
            role = turn.get("role", "")
            text = turn.get("text", "")
            dialogue_text += f"{role}: {text}\n"

        # 슬롯 목록 구성
        slot_info = []
        for domain, slots in self.SLOTS.items():
            # "도메인-슬롯" 형식에서 슬롯 이름만 추출
            slot_names = [s.split("-", 1)[1] for s in slots]
            slot_info.append(f"{domain}: {', '.join(slot_names)}")
        slots_text = "\n".join(slot_info)

        prompt = f"""다음 대화에서 사용자의 요청사항을 대화 상태(dialogue state)로 추출하세요.

대화:
{dialogue_text}

도메인별 가능한 슬롯:
{slots_text}

출력 규칙:
- 각 슬롯을 "도메인-슬롯-값" 형태로, 쉼표로 구분하여 나열
- 반드시 위에 정의된 슬롯 이름만 사용
- 예/아니오/있음/없음/가능 등은 모두 yes로 표기
- 숫자는 단위 없이 숫자만 (예: 3명 → 3, 2일 → 2)
- 시간은 HH:MM 형식 (예: 13:10)
- 가격대는 "저렴", "적당", "비싼" 중 선택
- 사용자가 "상관없다/아무거나" 등으로 말하면 dontcare로 표기

예시: 식당-지역-서울 중앙, 식당-종류-한식당, 식당-예약 명수-3, 식당-주차 가능-yes, 숙소-가격대-dontcare

대화 상태:"""
        return prompt

    def extract_answer(self, model_output: str) -> List[str]:
        """답변 추출 - 도메인-슬롯-값 형식 파싱"""
        output = model_output.strip()
        states = []

        # 쉼표 또는 줄바꿈으로 구분된 상태들 파싱
        # "도메인-슬롯-값" 패턴 찾기
        pattern = r'(관광|숙소|식당|지하철|택시)-([^,\n-]+)-([^,\n]+)'
        matches = re.findall(pattern, output)

        for domain, slot, value in matches:
            state = f"{domain}-{slot.strip()}-{value.strip()}"
            if state not in states:
                states.append(state)

        return sorted(states)

    def get_reference(self, example: Dict[str, Any]) -> List[str]:
        """정답 추출 - 마지막 user 턴에서 state 가져오기"""
        dialogue = example.get("dialogue", [])

        # 마지막 user 턴에서 state 추출 (sys 턴은 state가 비어있음)
        for turn in reversed(dialogue):
            if turn.get("role") == "user":
                state = turn.get("state", [])
                return sorted(state) if state else []

        return []

    def evaluate(
        self,
        predictions: List[List[str]],
        references: List[List[str]],
    ) -> Dict[str, float]:
        """JGA(Joint Goal Accuracy)와 Slot F1 계산"""
        jga_correct = 0
        total = len(predictions)

        # Slot-level metrics
        total_tp = 0  # True Positives
        total_fp = 0  # False Positives
        total_fn = 0  # False Negatives

        for pred, ref in zip(predictions, references):
            pred_set = set(pred) if pred else set()
            ref_set = set(ref) if ref else set()

            # JGA: 완전 일치
            if pred_set == ref_set:
                jga_correct += 1

            # Slot F1 계산을 위한 TP, FP, FN
            tp = len(pred_set & ref_set)
            fp = len(pred_set - ref_set)
            fn = len(ref_set - pred_set)

            total_tp += tp
            total_fp += fp
            total_fn += fn

        jga = jga_correct / total if total > 0 else 0

        # Slot Precision, Recall, F1
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        slot_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "joint_goal_accuracy": jga,
            "slot_precision": precision,
            "slot_recall": recall,
            "slot_f1": slot_f1,
        }

    def get_system_prompt(self) -> str:
        return "당신은 대화 상태 추적 전문가입니다. 대화에서 사용자의 요청을 '도메인-슬롯-값' 형식으로 추출하세요."
