import json
import re
from pathlib import Path

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


def analyze_task(filepath, choice_labels=['A', 'B', 'C', 'D']):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    raw_outputs = data['raw_outputs']
    total = len(raw_outputs)

    stored_correct = 0
    recalc_correct = 0
    mismatches = 0

    for item in raw_outputs:
        model_output = item.get('model_output', '')
        stored_pred = item['prediction']
        reference = item['reference']

        if model_output and isinstance(reference, str) and reference in choice_labels:
            recalc_pred = mc_extract_answer(model_output, choice_labels)
        else:
            recalc_pred = stored_pred

        if stored_pred == reference:
            stored_correct += 1
        if recalc_pred == reference:
            recalc_correct += 1
        if stored_pred != recalc_pred:
            mismatches += 1

    return {
        'stored_acc': stored_correct / total * 100,
        'recalc_acc': recalc_correct / total * 100,
        'mismatches': mismatches,
        'total': total,
        'mismatch_pct': mismatches / total * 100
    }


# Nova 모델들의 HellaSwag와 KMMLU 확인
print("Nova 모델 HellaSwag/KMMLU 재검증")
print("=" * 70)

models = ['nova-lite', 'nova-pro', 'nova-2-lite']
tasks = [
    ('kobest_hellaswag', ['A', 'B', 'C', 'D']),
    ('kobest_copa', ['A', 'B']),
    ('kmmlu_all', ['A', 'B', 'C', 'D']),
]

results_dir = Path('results/raw')

for model in models:
    print(f"\n{model}:")
    for task, labels in tasks:
        pattern = f"{model}_{task}_*.json"
        files = list(results_dir.glob(pattern))
        if files:
            filepath = files[0]
            result = analyze_task(filepath, labels)
            diff = result['recalc_acc'] - result['stored_acc']
            flag = "<<--" if abs(diff) > 1 else ""
            print(f"  {task}: stored={result['stored_acc']:.1f}% recalc={result['recalc_acc']:.1f}% mismatch={result['mismatch_pct']:.1f}% {flag}")
