"""
KLUE DP 벤치마크
Dependency Parsing - 의존 구문 분석
UAS/LAS 지표를 사용한 정확한 평가
"""
from typing import Dict, Any, List, Tuple
import re

from datasets import Dataset

from src.benchmarks.base_benchmark import BaseBenchmark
from src.data.dataset_loader import dataset_loader


class DPBenchmark(BaseBenchmark):
    """KLUE DP 벤치마크"""

    # 주요 deprel 태그 설명 (프롬프트에 사용)
    DEPREL_TAGS = {
        "NP": "명사구",
        "NP_SBJ": "주어",
        "NP_OBJ": "목적어",
        "NP_MOD": "명사 수식어",
        "NP_AJT": "부사격 명사구",
        "NP_CNJ": "접속 명사구",
        "NP_CMP": "보어",
        "VP": "동사구 (서술어)",
        "VP_MOD": "동사 수식어",
        "VP_AJT": "부사격 동사구",
        "VP_SBJ": "주어 동사구",
        "VP_OBJ": "목적어 동사구",
        "VNP": "서술격 조사구",
        "AP": "부사구",
        "DP": "관형사구",
        "IP": "감탄사구",
        "X": "기타",
        "L": "왼쪽 부가어",
        "R": "오른쪽 부가어",
    }

    def __init__(self):
        super().__init__(
            benchmark_name="klue",
            task_name="dp",
        )

    def load_dataset(self, split: str = "test") -> Dataset:
        """데이터셋 로드"""
        return dataset_loader.load_klue("dp", split)

    def format_prompt(self, example: Dict[str, Any]) -> str:
        """프롬프트 포맷 - 의존 구문 분석 요청"""
        word_form = example.get("word_form", [])

        # 단어 목록 구성 (인덱스와 함께)
        words_text = ""
        for i, word in enumerate(word_form):
            words_text += f"{i+1}. {word}\n"

        # 주요 의존 관계 태그 설명
        tag_desc = "\n".join([f"- {tag}: {desc}" for tag, desc in list(self.DEPREL_TAGS.items())[:10]])

        prompt = f"""다음 문장의 의존 구문 분석을 수행하세요.

문장의 단어 목록:
{words_text}
주요 의존 관계 유형:
{tag_desc}

출력 규칙:
- 각 단어에 대해 "단어번호:머리번호:의존관계" 형태로 출력
- 줄바꿈으로 구분
- 머리 인덱스 0은 ROOT(최상위 서술어)
- 설명 없이 결과만 출력

예시:
1:2:NP_MOD
2:5:NP_SBJ
3:4:NP_MOD
4:5:NP_OBJ
5:0:VP

결과:"""
        return prompt

    def extract_answer(self, model_output: str) -> List[Tuple[int, int, str]]:
        """
        답변 추출 - (word_idx, head_idx, deprel) 튜플 리스트 파싱

        Returns:
            List[Tuple[int, int, str]]: [(단어인덱스, 머리인덱스, 의존관계), ...]
        """
        output = model_output.strip()
        results = []

        # "번호:머리:관계" 패턴 찾기
        # 예: "1:2:NP_SBJ" 또는 "1: 2: NP_SBJ" 또는 "1 : 2 : NP_SBJ"
        pattern = r'(\d+)\s*:\s*(\d+)\s*:\s*([A-Z_]+)'
        matches = re.findall(pattern, output)

        for word_idx, head_idx, deprel in matches:
            results.append((int(word_idx), int(head_idx), deprel.strip()))

        return results

    def get_reference(self, example: Dict[str, Any]) -> List[Tuple[int, int, str]]:
        """
        정답 추출 - (word_idx, head_idx, deprel) 튜플 리스트

        Returns:
            List[Tuple[int, int, str]]: [(단어인덱스, 머리인덱스, 의존관계), ...]
        """
        head = example.get("head", [])
        deprel = example.get("deprel", [])

        results = []
        for i, (h, d) in enumerate(zip(head, deprel)):
            results.append((i + 1, h, d))

        return results

    def evaluate(
        self,
        predictions: List[List[Tuple[int, int, str]]],
        references: List[List[Tuple[int, int, str]]],
    ) -> Dict[str, float]:
        """
        UAS/LAS 계산

        UAS (Unlabeled Attachment Score): head만 맞으면 정답
        LAS (Labeled Attachment Score): head와 deprel 둘 다 맞아야 정답
        """
        total_tokens = 0
        uas_correct = 0
        las_correct = 0

        valid_sentences = 0
        complete_sentences = 0

        for pred, ref in zip(predictions, references):
            if not pred or not ref:
                continue

            valid_sentences += 1

            # 예측을 딕셔너리로 변환 (word_idx -> (head, deprel))
            pred_dict = {p[0]: (p[1], p[2]) for p in pred}

            sentence_uas_correct = 0
            sentence_total = 0

            for ref_word_idx, ref_head, ref_deprel in ref:
                total_tokens += 1
                sentence_total += 1

                if ref_word_idx in pred_dict:
                    pred_head, pred_deprel = pred_dict[ref_word_idx]

                    # UAS: head만 비교
                    if pred_head == ref_head:
                        uas_correct += 1
                        sentence_uas_correct += 1

                        # LAS: head와 deprel 둘 다 비교
                        if pred_deprel == ref_deprel:
                            las_correct += 1

            # 문장 전체가 맞았는지 확인
            if sentence_total > 0 and sentence_uas_correct == sentence_total:
                complete_sentences += 1

        # 메트릭 계산
        uas = uas_correct / total_tokens if total_tokens > 0 else 0
        las = las_correct / total_tokens if total_tokens > 0 else 0
        sentence_accuracy = complete_sentences / valid_sentences if valid_sentences > 0 else 0

        return {
            "uas": uas,
            "las": las,
            "sentence_accuracy": sentence_accuracy,
            "total_tokens": total_tokens,
            "valid_sentences": valid_sentences,
        }

    def get_system_prompt(self) -> str:
        return "당신은 한국어 의존 구문 분석 전문가입니다. 문장의 각 단어에 대해 의존 관계(head와 deprel)를 정확하게 분석하세요."
