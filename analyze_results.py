import json
import re

with open('results/raw/claude-sonnet-4-5_kobest_hellaswag_n50.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

choice_labels = ['A', 'B', 'C', 'D']

def extract_answer(model_output):
    output = model_output.strip()
    output_upper = output.upper()
    labels_pattern = '|'.join(choice_labels)

    if output_upper in choice_labels:
        return output_upper, 'direct'

    match = re.search(rf'^[\s]*({labels_pattern})[.\):\s]', output_upper, re.MULTILINE)
    if match:
        return match.group(1), 'pattern4'

    match = re.search(rf'\b({labels_pattern})\b', output_upper)
    if match:
        return match.group(1), 'pattern5'

    return choice_labels[0], 'default'

print('=== HellaSwag 결과 분석 ===')
errors = 0
for i, item in enumerate(data['raw_outputs'][:10]):
    model_output = item['model_output']
    stored_pred = item['prediction']
    reference = item['reference']

    recalc_pred, pattern = extract_answer(model_output)

    if stored_pred != recalc_pred:
        errors += 1
        print(f'[{i+1}] MISMATCH!')
        print(f'  output start: {repr(model_output[:60])}')
        print(f'  stored: {stored_pred}, recalc: {recalc_pred} ({pattern}), ref: {reference}')
    else:
        correct = 'O' if stored_pred == reference else 'X'
        print(f'[{i+1}] OK - pred={stored_pred}, ref={reference}, correct={correct}')

print(f'\nMismatches in first 10: {errors}')

# WIC 분석
print('\n\n=== WiC 결과 분석 ===')
with open('results/raw/claude-sonnet-4-5_kobest_wic_n50.json', 'r', encoding='utf-8') as f:
    wic_data = json.load(f)

def extract_bool(model_output):
    output = model_output.strip().lower()

    # 첫 단어만 확인
    first_word = output.split()[0] if output.split() else ''
    if first_word in ['true', 'false']:
        return first_word == 'true', 'first_word'

    positive_patterns = ["true", "yes", "예", "맞", "참", "o", "1"]
    for pattern in positive_patterns:
        if pattern in output:
            return True, f'pattern:{pattern}'

    return False, 'default'

errors = 0
for i, item in enumerate(wic_data['raw_outputs'][:10]):
    model_output = item['model_output']
    stored_pred = item['prediction']
    reference = item['reference']

    recalc_pred, pattern = extract_bool(model_output)

    if stored_pred != recalc_pred:
        errors += 1
        print(f'[{i+1}] MISMATCH!')
        print(f'  output start: {repr(model_output[:80])}')
        print(f'  stored: {stored_pred}, recalc: {recalc_pred} ({pattern}), ref: {reference}')
    else:
        correct = 'O' if stored_pred == reference else 'X'
        print(f'[{i+1}] OK - pred={stored_pred}, ref={reference}, correct={correct}')

print(f'\nMismatches in first 10: {errors}')
