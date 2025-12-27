"""HAE-RAE 결과 검증 스크립트"""
import json
import re
from pathlib import Path

def mc_extract_answer(model_output, choice_labels):
    output = model_output.strip()
    output_upper = output.upper()

    if output_upper in choice_labels:
        return output_upper

    labels_pattern = '|'.join(choice_labels)

    first_line = output_upper.split('\n')[0].strip()
    if first_line in choice_labels:
        return first_line

    match = re.search(r'정답[:\s]*[\(\[]?(' + labels_pattern + r')[\)\]]?', output_upper)
    if match:
        return match.group(1)

    match = re.search(r'\*\*[\(\[]?(' + labels_pattern + r')[\)\]]?\*\*', output_upper)
    if match:
        return match.group(1)

    match = re.search(r'[\(\[](' + labels_pattern + r')[\)\]]', output_upper)
    if match:
        return match.group(1)

    match = re.search(r'^[\s]*(' + labels_pattern + r')[.\):\s]', output_upper, re.MULTILINE)
    if match:
        return match.group(1)

    match = re.search(r'\b(' + labels_pattern + r')\b', output_upper)
    if match:
        return match.group(1)

    return choice_labels[0]

results_dir = Path('results/raw')
choice_labels = ['A', 'B', 'C', 'D', 'E']

models = ['claude-sonnet-4-5', 'nova-lite', 'nova-pro', 'nova-2-lite']

for model in models:
    print(f'\n{"=" * 70}')
    print(f'{model} HAE-RAE')
    print('=' * 70)

    for f in sorted(results_dir.glob(f'{model}_haerae_*.json')):
        with open(f, 'r', encoding='utf-8') as fp:
            data = json.load(fp)

        task = data['task_name']
        raw = data['raw_outputs']

        stored_correct = 0
        recalc_correct = 0
        mismatch = 0

        for item in raw:
            model_output = item.get('model_output', '')
            stored_pred = item['prediction']
            ref = item['reference']

            if model_output:
                recalc_pred = mc_extract_answer(model_output, choice_labels)
            else:
                recalc_pred = stored_pred

            if stored_pred == ref:
                stored_correct += 1
            if recalc_pred == ref:
                recalc_correct += 1
            if stored_pred != recalc_pred:
                mismatch += 1

        total = len(raw)
        stored_acc = stored_correct / total * 100
        recalc_acc = recalc_correct / total * 100
        diff = recalc_acc - stored_acc
        flag = 'WARNING' if abs(diff) > 1 else 'OK'

        print(f'{task:<25} stored: {stored_acc:5.1f}%  recalc: {recalc_acc:5.1f}%  mismatch: {mismatch:3d}/{total} [{flag}]')
