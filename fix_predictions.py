"""
extract_answer 버그로 인해 잘못 저장된 prediction 재계산 스크립트
- raw_outputs의 model_output에서 prediction 재추출
- metrics 재계산
- 파일 업데이트
"""
import json
import re
from pathlib import Path
from datetime import datetime


def mc_extract_answer(model_output, choice_labels):
    """MultipleChoiceBenchmark의 수정된 extract_answer"""
    output = model_output.strip()
    output_upper = output.upper()

    # 직접 라벨인 경우
    if output_upper in choice_labels:
        return output_upper

    labels_pattern = "|".join(choice_labels)

    # 1. "정답: (A)" 또는 "정답: A" 패턴
    match = re.search(rf'정답[:\s]*[\(\[]?({labels_pattern})[\)\]]?', output_upper)
    if match:
        return match.group(1)

    # 2. "**(A)**" 또는 "**A**" 패턴 (마크다운 볼드)
    match = re.search(rf'\*\*[\(\[]?({labels_pattern})[\)\]]?\*\*', output_upper)
    if match:
        return match.group(1)

    # 3. "(A)" 또는 "[A]" 패턴
    match = re.search(rf'[\(\[]({labels_pattern})[\)\]]', output_upper)
    if match:
        return match.group(1)

    # 4. "A." 또는 "A)" 또는 "A:" 패턴 (문장 시작)
    match = re.search(rf'^[\s]*({labels_pattern})[.\):\s]', output_upper, re.MULTILINE)
    if match:
        return match.group(1)

    # 5. 단독 레이블
    match = re.search(rf'\b({labels_pattern})\b', output_upper)
    if match:
        return match.group(1)

    # 추출 실패 시 첫 번째 선택지 반환
    return choice_labels[0]


def calculate_accuracy(predictions, references):
    """정확도 계산"""
    if not predictions or not references:
        return 0.0
    correct = sum(p == r for p, r in zip(predictions, references))
    return correct / len(predictions)


def fix_result_file(filepath, choice_labels):
    """결과 파일 수정"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    task = data['task_name']
    benchmark = data['benchmark_name']
    model = data['model_name']

    # 원본 metrics 저장
    original_metrics = data['metrics'].copy()

    # raw_outputs에서 prediction 재계산
    predictions = []
    references = []
    fixed_count = 0

    for item in data['raw_outputs']:
        model_output = item.get('model_output', '')
        reference = item['reference']
        old_pred = item['prediction']

        if model_output:
            new_pred = mc_extract_answer(model_output, choice_labels)
        else:
            new_pred = old_pred

        # prediction 업데이트
        if old_pred != new_pred:
            fixed_count += 1
            item['prediction'] = new_pred

        predictions.append(new_pred)
        references.append(reference)

    # metrics 재계산
    new_accuracy = calculate_accuracy(predictions, references)
    data['metrics']['accuracy'] = new_accuracy

    # metadata에 수정 정보 추가
    data['metadata']['fixed_at'] = datetime.now().isoformat()
    data['metadata']['original_accuracy'] = original_metrics.get('accuracy', 0)
    data['metadata']['predictions_fixed'] = fixed_count

    # 파일 저장
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return {
        'file': filepath.name,
        'model': model,
        'task': f"{benchmark}/{task}",
        'original_acc': original_metrics.get('accuracy', 0) * 100,
        'new_acc': new_accuracy * 100,
        'fixed_count': fixed_count,
        'total': len(predictions)
    }


def main():
    results_dir = Path('results/raw')

    # 수정할 태스크 목록 (태스크명, 선택지)
    tasks_to_fix = [
        ('kmmlu_all', ['A', 'B', 'C', 'D']),
        ('kobest_hellaswag', ['A', 'B', 'C', 'D']),
        ('kobest_copa', ['A', 'B']),
        ('klue_ynat', ['A', 'B', 'C', 'D', 'E', 'F', 'G']),  # 7개 카테고리
    ]

    # 모든 모델
    models = ['claude-sonnet-4-5', 'nova-lite', 'nova-pro', 'nova-2-lite']

    print("=" * 80)
    print("extract_answer 버그 수정")
    print("=" * 80)

    results = []

    for model in models:
        for task, labels in tasks_to_fix:
            pattern = f"{model}_{task}_*.json"
            files = list(results_dir.glob(pattern))

            for filepath in files:
                result = fix_result_file(filepath, labels)
                results.append(result)

                diff = result['new_acc'] - result['original_acc']
                if abs(diff) > 0.1:
                    print(f"\n[FIXED] {result['file']}")
                    print(f"  {result['original_acc']:.1f}% -> {result['new_acc']:.1f}% ({result['fixed_count']}/{result['total']} predictions fixed)")

    print("\n" + "=" * 80)
    print("수정 완료 요약")
    print("=" * 80)

    significant_fixes = [r for r in results if abs(r['new_acc'] - r['original_acc']) > 1]
    print(f"\n총 {len(results)}개 파일 처리, {len(significant_fixes)}개 파일에서 유의미한 변화")

    if significant_fixes:
        print(f"\n{'Model':<20} {'Task':<25} {'Before':>10} {'After':>10} {'Diff':>10}")
        print("-" * 80)
        for r in significant_fixes:
            diff = r['new_acc'] - r['original_acc']
            print(f"{r['model']:<20} {r['task']:<25} {r['original_acc']:>9.1f}% {r['new_acc']:>9.1f}% {diff:>+9.1f}%")


if __name__ == "__main__":
    main()
