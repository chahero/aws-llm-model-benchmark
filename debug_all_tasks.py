import json
import re
import os
from pathlib import Path

# MultipleChoiceBenchmark의 extract_answer 로직
def mc_extract_answer(model_output, choice_labels=['A', 'B', 'C', 'D']):
    output = model_output.strip()
    output_upper = output.upper()

    if output_upper in choice_labels:
        return output_upper

    labels_pattern = "|".join(choice_labels)

    match = re.search(rf'정답[:\s]*[\(\[]?({labels_pattern})[\)\]]?', output_upper)
    if match:
        return match.group(1)

    match = re.search(rf'\*\*[\(\[]?({labels_pattern})[\)\]]?\*\*', output_upper)
    if match:
        return match.group(1)

    match = re.search(rf'[\(\[]({labels_pattern})[\)\]]', output_upper)
    if match:
        return match.group(1)

    match = re.search(rf'^[\s]*({labels_pattern})[.\):\s]', output_upper, re.MULTILINE)
    if match:
        return match.group(1)

    match = re.search(rf'\b({labels_pattern})\b', output_upper)
    if match:
        return match.group(1)

    return choice_labels[0]


# BinaryClassificationBenchmark의 extract_answer 로직
def bin_extract_answer(model_output):
    output = model_output.strip().lower()
    positive_patterns = ["true", "yes", "예", "맞", "참", "o", "1"]
    for pattern in positive_patterns:
        if pattern in output:
            return True
    return False


# 분류 태스크 분석
def analyze_classification_task(filepath, num_choices=None):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    task_name = data['task_name']
    raw_outputs = data['raw_outputs']
    total = len(raw_outputs)

    # 태스크 유형 판별
    first_ref = raw_outputs[0]['reference'] if raw_outputs else None

    if isinstance(first_ref, bool):
        # Binary classification
        extract_fn = bin_extract_answer
    elif isinstance(first_ref, str) and first_ref in ['A', 'B', 'C', 'D', 'E']:
        # Multiple choice
        if num_choices:
            choice_labels = ['A', 'B', 'C', 'D', 'E'][:num_choices]
        else:
            # 추정
            refs = set(item['reference'] for item in raw_outputs if isinstance(item.get('reference'), str))
            choice_labels = sorted(list(refs))
        extract_fn = lambda x: mc_extract_answer(x, choice_labels)
    else:
        return None  # 다른 유형

    # 재계산
    stored_correct = 0
    recalc_correct = 0
    mismatches = 0

    for item in raw_outputs:
        model_output = item.get('model_output', '')
        stored_pred = item['prediction']
        reference = item['reference']

        if model_output:
            recalc_pred = extract_fn(model_output)
        else:
            recalc_pred = stored_pred

        if stored_pred == reference:
            stored_correct += 1
        if recalc_pred == reference:
            recalc_correct += 1
        if stored_pred != recalc_pred:
            mismatches += 1

    return {
        'task': task_name,
        'total': total,
        'stored_acc': stored_correct / total * 100,
        'recalc_acc': recalc_correct / total * 100,
        'mismatches': mismatches,
        'mismatch_pct': mismatches / total * 100
    }


# Claude Sonnet 4.5 결과 분석 (HAE-RAE 포함)
print("="*70)
print("Claude Sonnet 4.5 전체 태스크 재검증 (HAE-RAE 포함)")
print("="*70)

results_dir = Path('results/raw')
files = sorted(results_dir.glob('claude-sonnet-4-5_*.json'))

results = []
for filepath in files:
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    task = data['task_name']
    benchmark = data['benchmark_name']

    # 태스크별 선택지 수
    num_choices_map = {
        'hellaswag': 4,
        'copa': 2,
        'boolq': 2,  # binary이지만 A/B로 처리
        'sentineg': 2,
        'wic': None,  # binary
        'nli': 3,
        'ynat': 7,
        're': None,  # 다양
    }

    result = analyze_classification_task(filepath, num_choices_map.get(task))
    if result:
        result['benchmark'] = benchmark
        results.append(result)

# 결과 출력
print(f"\n{'태스크':20s} | {'저장 정확도':>12s} | {'재계산 정확도':>12s} | {'불일치':>8s}")
print("-" * 70)

problem_tasks = []
for r in results:
    diff = r['recalc_acc'] - r['stored_acc']
    flag = "<<--" if abs(diff) > 1 else ""
    print(f"{r['benchmark']}/{r['task']:12s} | {r['stored_acc']:>10.1f}% | {r['recalc_acc']:>10.1f}% | {r['mismatches']:>4d} ({r['mismatch_pct']:.0f}%) {flag}")

    if r['mismatch_pct'] > 5:
        problem_tasks.append(r)

print("\n" + "="*70)
print("문제 있는 태스크 (불일치 > 5%)")
print("="*70)
for r in problem_tasks:
    print(f"- {r['benchmark']}/{r['task']}: 불일치 {r['mismatches']}/{r['total']} ({r['mismatch_pct']:.0f}%)")
    print(f"  저장: {r['stored_acc']:.1f}% → 재계산: {r['recalc_acc']:.1f}%")
