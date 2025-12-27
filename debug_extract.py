import json
import re

# MultipleChoiceBenchmark의 extract_answer 로직 복사
def mc_extract_answer(model_output, choice_labels=['A', 'B', 'C', 'D']):
    output = model_output.strip()
    output_upper = output.upper()

    # 직접 라벨인 경우 (출력이 단순히 "A", "B" 등)
    if output_upper in choice_labels:
        return output_upper

    # 패턴 기반 추출 (우선순위 순)
    labels_pattern = "|".join(choice_labels)

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
    return choice_labels[0]


# BinaryClassificationBenchmark의 extract_answer 로직 복사
def bin_extract_answer(model_output):
    output = model_output.strip().lower()

    # True 패턴 매칭
    positive_patterns = ["true", "yes", "예", "맞", "참", "o", "1"]
    for pattern in positive_patterns:
        if pattern in output:
            return True

    return False


# HellaSwag 테스트
print("="*60)
print("HellaSwag 테스트 (MultipleChoiceBenchmark)")
print("="*60)

with open('results/raw/claude-sonnet-4-5_kobest_hellaswag_n50.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

errors = 0
for i, item in enumerate(data['raw_outputs']):
    model_output = item['model_output']
    stored_pred = item['prediction']
    reference = item['reference']

    # 현재 코드로 추출
    current_pred = mc_extract_answer(model_output)

    if stored_pred != current_pred:
        errors += 1
        if errors <= 5:  # 처음 5개만 출력
            print(f"\n[{i+1}] 불일치 발견!")
            print(f"  model_output 첫줄: {model_output.split(chr(10))[0]}")
            print(f"  저장된 값: {stored_pred}")
            print(f"  현재 코드: {current_pred}")
            print(f"  정답: {reference}")

total = len(data['raw_outputs'])
print(f"\n총 {total}개 중 {errors}개 불일치 ({errors/total*100:.1f}%)")

# WiC 테스트
print("\n" + "="*60)
print("WiC 테스트 (BinaryClassificationBenchmark)")
print("="*60)

with open('results/raw/claude-sonnet-4-5_kobest_wic_n50.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

errors = 0
for i, item in enumerate(data['raw_outputs']):
    model_output = item['model_output']
    stored_pred = item['prediction']
    reference = item['reference']

    current_pred = bin_extract_answer(model_output)

    if stored_pred != current_pred:
        errors += 1
        if errors <= 5:
            print(f"\n[{i+1}] 불일치 발견!")
            print(f"  model_output 첫줄: {model_output.split(chr(10))[0]}")
            print(f"  저장된 값: {stored_pred}")
            print(f"  현재 코드: {current_pred}")
            print(f"  정답: {reference}")

total = len(data['raw_outputs'])
print(f"\n총 {total}개 중 {errors}개 불일치 ({errors/total*100:.1f}%)")

# 현재 코드로 재평가
print("\n" + "="*60)
print("현재 코드로 재평가 시 정확도")
print("="*60)

# HellaSwag
with open('results/raw/claude-sonnet-4-5_kobest_hellaswag_n50.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

stored_correct = sum(1 for item in data['raw_outputs'] if item['prediction'] == item['reference'])
recalc_correct = sum(1 for item in data['raw_outputs'] if mc_extract_answer(item['model_output']) == item['reference'])
total = len(data['raw_outputs'])

print(f"HellaSwag:")
print(f"  저장된 정확도: {stored_correct}/{total} = {stored_correct/total*100:.1f}%")
print(f"  재계산 정확도: {recalc_correct}/{total} = {recalc_correct/total*100:.1f}%")

# WiC
with open('results/raw/claude-sonnet-4-5_kobest_wic_n50.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

stored_correct = sum(1 for item in data['raw_outputs'] if item['prediction'] == item['reference'])
recalc_correct = sum(1 for item in data['raw_outputs'] if bin_extract_answer(item['model_output']) == item['reference'])
total = len(data['raw_outputs'])

print(f"WiC:")
print(f"  저장된 정확도: {stored_correct}/{total} = {stored_correct/total*100:.1f}%")
print(f"  재계산 정확도: {recalc_correct}/{total} = {recalc_correct/total*100:.1f}%")
