"""
KLUE NER 데이터 디버깅 스크립트
"""
from datasets import load_dataset

# KLUE NER 태그 매핑
TAG_MAP = {
    0: "O",
    1: "B-PS", 2: "I-PS",
    3: "B-LC", 4: "I-LC",
    5: "B-OG", 6: "I-OG",
    7: "B-DT", 8: "I-DT",
    9: "B-TI", 10: "I-TI",
    11: "B-QT", 12: "I-QT",
}

def analyze_example(example, idx):
    """샘플 분석"""
    tokens = example.get("tokens", [])
    ner_tags = example.get("ner_tags", [])

    print(f"\n{'='*60}")
    print(f"Example {idx}")
    print(f"{'='*60}")

    # 원문 복원
    sentence = "".join(tokens)
    print(f"원문: {sentence[:100]}...")

    # 토큰 개수
    print(f"토큰 수: {len(tokens)}, 태그 수: {len(ner_tags)}")

    # 처음 20개 토큰과 태그
    print(f"\n처음 20개 토큰-태그:")
    for i in range(min(20, len(tokens))):
        tag_str = TAG_MAP.get(ner_tags[i], f"UNK({ner_tags[i]})")
        print(f"  [{i}] '{tokens[i]}' -> {tag_str}")

    # 엔티티 추출 (기존 방식)
    entities = []
    current_entity = []
    current_type = None

    for token, tag_id in zip(tokens, ner_tags):
        tag = TAG_MAP.get(tag_id, "O")

        if tag.startswith("B-"):
            if current_entity:
                entities.append({
                    "entity": "".join(current_entity),  # 공백 없이 합침
                    "type": current_type
                })
            current_entity = [token]
            current_type = tag[2:]
        elif tag.startswith("I-") and current_entity:
            current_entity.append(token)
        else:
            if current_entity:
                entities.append({
                    "entity": "".join(current_entity),  # 공백 없이 합침
                    "type": current_type
                })
            current_entity = []
            current_type = None

    if current_entity:
        entities.append({
            "entity": "".join(current_entity),
            "type": current_type
        })

    print(f"\n추출된 엔티티 ({len(entities)}개):")
    for e in entities[:10]:
        print(f"  - [{e['type']}] {e['entity']}")

    return entities


def main():
    print("KLUE NER 데이터셋 로드 중...")

    # validation split 로드 (test는 비공개)
    try:
        ds = load_dataset("klue", "ner", split="validation")
    except:
        ds = load_dataset("klue", "ner", split="train")

    print(f"로드된 예제 수: {len(ds)}")
    print(f"컬럼: {ds.column_names}")

    # 처음 5개 분석
    for i in range(min(5, len(ds))):
        analyze_example(ds[i], i)

    # 태그 분포 확인
    print(f"\n{'='*60}")
    print("전체 태그 분포 (처음 1000개 예제)")
    print(f"{'='*60}")

    tag_counts = {}
    for i in range(min(1000, len(ds))):
        for tag_id in ds[i]["ner_tags"]:
            tag_str = TAG_MAP.get(tag_id, f"UNK({tag_id})")
            tag_counts[tag_str] = tag_counts.get(tag_str, 0) + 1

    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
        print(f"  {tag}: {count}")


if __name__ == "__main__":
    main()
