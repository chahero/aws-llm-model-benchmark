"""수정된 extract_answer 로직 테스트"""
import re

# MultipleChoiceBenchmark의 수정된 extract_answer
def mc_extract_answer(model_output, choice_labels=['A', 'B', 'C', 'D']):
    output = model_output.strip()
    output_upper = output.upper()

    if output_upper in choice_labels:
        return output_upper

    labels_pattern = "|".join(choice_labels)

    # 0. 첫 줄이 단일 라벨인 경우 (예: "A\n\n설명...")
    first_line = output_upper.split('\n')[0].strip()
    if first_line in choice_labels:
        return first_line

    # 1. "정답: (A)" 패턴
    match = re.search(rf'정답[:\s]*[\(\[]?({labels_pattern})[\)\]]?', output_upper)
    if match:
        return match.group(1)

    # 2. 마크다운 볼드
    match = re.search(rf'\*\*[\(\[]?({labels_pattern})[\)\]]?\*\*', output_upper)
    if match:
        return match.group(1)

    # 3. 괄호 패턴
    match = re.search(rf'[\(\[]({labels_pattern})[\)\]]', output_upper)
    if match:
        return match.group(1)

    # 4. 문장 시작
    match = re.search(rf'^[\s]*({labels_pattern})[.\):\s]', output_upper, re.MULTILINE)
    if match:
        return match.group(1)

    # 5. 단독 레이블
    match = re.search(rf'\b({labels_pattern})\b', output_upper)
    if match:
        return match.group(1)

    return choice_labels[0]


# BinaryClassificationBenchmark의 수정된 extract_answer
def bin_extract_answer(model_output):
    output = model_output.strip().lower()

    # 1. 첫 단어 확인
    first_word = output.split()[0] if output.split() else ""
    first_word_clean = re.sub(r'[^a-z가-힣]', '', first_word)

    if first_word_clean in ["true", "yes", "예", "맞습니다", "맞음", "참"]:
        return True
    if first_word_clean in ["false", "no", "아니오", "아니요", "틀립니다", "틀림", "거짓"]:
        return False

    # 2. 첫 줄 확인
    first_line = output.split('\n')[0].strip().lower()
    if first_line in ["true", "false", "yes", "no"]:
        return first_line in ["true", "yes"]

    # 3. 단어 경계 패턴
    if re.search(r'\btrue\b', output):
        return True
    if re.search(r'\bfalse\b', output):
        return False

    # 4. 한국어 (문장 시작)
    if re.match(r'^(예|맞습니다|맞음|참)', output):
        return True
    if re.match(r'^(아니오|아니요|틀립니다|틀림|거짓)', output):
        return False

    return False


# 테스트 케이스
print("=" * 60)
print("MultipleChoiceBenchmark 테스트")
print("=" * 60)

mc_tests = [
    ("D\n\n문맥에서 여자가...", "D"),
    ("A", "A"),
    ("정답: B", "B"),
    ("**C**가 정답입니다", "C"),
    ("(A) 첫 번째 선택지가...", "A"),
    ("저는 D라고 생각합니다", "D"),
]

for output, expected in mc_tests:
    result = mc_extract_answer(output)
    status = "OK" if result == expected else "FAIL"
    print(f"[{status}] '{output[:30]}...' -> {result} (expected: {expected})")


print("\n" + "=" * 60)
print("BinaryClassificationBenchmark 테스트")
print("=" * 60)

bin_tests = [
    ("True\n\n두 문장 모두...", True),
    ("False\n\n문장 1의 '풍경'은...", False),
    ("False\n\n문장 1의 '포수'는 사냥꾼을 의미하고...", False),
    ("예, 같은 의미입니다", True),
    ("아니오, 다른 의미입니다", False),
    ("이것은 맞지 않습니다", False),  # "맞"이 포함되어도 False여야 함
    ("True", True),
    ("False", False),
]

for output, expected in bin_tests:
    result = bin_extract_answer(output)
    status = "OK" if result == expected else "FAIL"
    print(f"[{status}] '{output[:40]}...' -> {result} (expected: {expected})")
