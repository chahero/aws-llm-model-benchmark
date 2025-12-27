# AWS Bedrock Korean LLM Benchmark

AWS Bedrock 기반 LLM 모델의 한국어 성능을 평가하는 벤치마크 프레임워크입니다.

[English](README_EN.md) | **한국어**

## 개요

이 프로젝트는 AWS Bedrock에서 제공하는 Foundation Model(Nova 시리즈)의 한국어 처리 능력을 다양한 벤치마크로 평가합니다.

### 주요 기능

- **4개 표준 벤치마크 지원**: KLUE, KoBEST, HAE-RAE, KMMLU
- **27개+ 태스크**: 추론, 유사도, 개체명 인식, 독해, 분류 등
- **선택적 태스크 실행**: 특정 태스크만 재실행 가능 (`--tasks` 옵션)
- **자동 결과 저장**: JSON 및 마크다운 리포트 생성
- **웹 대시보드**: Streamlit 기반 시각화
- **비용 추적**: 토큰 사용량 및 예상 비용 계산
- **중복 실행 방지**: 동일 조건 테스트 자동 스킵

## 지원 모델

| 모델 | Model ID | 입력 (1K 토큰) | 출력 (1K 토큰) |
|------|----------|---------------|---------------|
| nova-lite | us.amazon.nova-lite-v1:0 | $0.00006 | $0.00024 |
| nova-pro | us.amazon.nova-pro-v1:0 | $0.0008 | $0.0032 |
| nova-2-lite | us.amazon.nova-2-lite-v1:0 | $0.00033 | $0.00275 |
| claude-sonnet-4-5 | global.anthropic.claude-sonnet-4-5-20250929-v1:0 | $0.003 | $0.015 |

> Model ID는 `.env` 파일에서 커스터마이즈 가능합니다.

## 벤치마크

### KLUE (Korean Language Understanding Evaluation)

| 태스크 | 설명 | 메트릭 |
|--------|------|--------|
| NLI | 자연어 추론 (함의/중립/모순) | Accuracy |
| STS | 문장 유사도 (0~5) | Pearson |
| NER | 개체명 인식 | F1 |
| RE | 관계 추출 | F1 |
| MRC | 기계 독해 | F1/EM |
| DP | 의존 구문 분석 | UAS/LAS |
| WoS | 대화 상태 추적 | JGA |
| YNAT | 뉴스 주제 분류 | Accuracy |

### KoBEST (Korean Balanced Evaluation of Significant Tasks)

| 태스크 | 설명 | 메트릭 |
|--------|------|--------|
| BoolQ | 예/아니오 질문 | Accuracy |
| COPA | 인과 추론 | Accuracy |
| WiC | 단어 문맥 의미 | Accuracy |
| HellaSwag | 상식 추론 | Accuracy |
| SentiNeg | 감성 부정 | Accuracy |

### HAE-RAE (Korean Cultural & Linguistic Benchmark)

| 태스크 | 설명 | 메트릭 |
|--------|------|--------|
| 표준어/맞춤법 | 올바른 표현 선택 | Accuracy |
| 외래어/로마자 | 외래어 표기법 | Accuracy |
| 희귀어/신조어 | 어휘 이해 | Accuracy |
| 일반 상식 | 한국 상식 | Accuracy |
| 역사 | 한국사 지식 | Accuracy |
| 독해 | 지문 이해 | Accuracy |

### KMMLU (Korean Massive Multitask Language Understanding)

| 태스크 | 설명 | 메트릭 |
|--------|------|--------|
| KMMLU | 45개 전문 분야 지식 평가 | Accuracy |

*회계, 법학, 의학, 공학, 역사, 수학, 물리학 등 45개 전문 분야 포함*

## 설치

### 요구 사항

- Python 3.9+
- AWS 계정 및 Bedrock 액세스 권한

### 설치 방법

```bash
# 저장소 클론
git clone https://github.com/your-repo/aws-llm-model-benchmark.git
cd aws-llm-model-benchmark

# 가상환경 생성
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 환경 설정
cp .env.example .env
# .env 파일에서 AWS 자격 증명 설정
```

## 사용법

### 기본 실행

```bash
# 전체 벤치마크 실행 (모든 모델, 모든 태스크)
python main.py

# 특정 모델만 테스트
python main.py --models nova-lite

# 특정 벤치마크만 테스트
python main.py --benchmarks klue

# 샘플 제한 (빠른 테스트용)
python main.py --limit 100

# 기존 결과 무시하고 재실행
python main.py --force

# 마크다운 리포트 생성
python main.py --report
```

### 선택적 태스크 실행

특정 태스크만 실행하여 시간을 절약할 수 있습니다:

```bash
# NER 태스크만 다시 실행 (기존 결과 덮어쓰기)
python main.py --models all --benchmarks klue --tasks ner --force

# 여러 태스크 선택 실행
python main.py --models all --benchmarks all --tasks ner boolq copa

# 특정 모델의 특정 태스크만 실행
python main.py --models nova-lite --benchmarks klue --tasks ner mrc --force
```

### 환경 변수 설정 (.env)

```bash
# AWS 설정
AWS_REGION=us-west-2
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret

# HuggingFace 토큰 (데이터셋 다운로드)
HF_TOKEN=your_hf_token

# 벤치마크 설정
BENCHMARK_MODELS=nova-lite,nova-pro,nova-2-lite
BENCHMARK_TASKS=klue,kobest,haerae
BENCHMARK_SAMPLE_LIMIT=        # 비워두면 전체 데이터
BENCHMARK_RATE_LIMIT=10.0
```

### 대시보드 실행

```bash
streamlit run dashboard/app.py
```

## 결과

### 저장 위치

```
results/
├── raw/                    # JSON 결과 파일
│   ├── nova-lite_klue_nli_n3000.json
│   └── ...
└── reports/                # 마크다운 리포트
    ├── nova-lite_klue_nli_n3000.md
    └── summary.md
```

### 파일명 형식

```
{모델}_{벤치마크}_{태스크}_n{샘플수}.json
{모델}_{벤치마크}_{태스크}_full.json  # 전체 데이터셋
```

### 결과 예시

```json
{
  "benchmark_name": "klue",
  "task_name": "nli",
  "model_name": "nova-lite",
  "metrics": {
    "accuracy": 0.808
  },
  "metadata": {
    "num_examples": 3000,
    "elapsed_time_sec": 4852,
    "cost_estimate": 0.19
  }
}
```

## 프로젝트 구조

```
aws-llm-model-benchmark/
├── config/
│   └── settings.py          # 설정 (모델 ID, 비용 등)
├── src/
│   ├── models/
│   │   ├── bedrock_client.py    # AWS Bedrock API
│   │   └── nova_models.py       # Nova 모델 래퍼
│   ├── benchmarks/
│   │   ├── klue/            # KLUE 8개 태스크
│   │   ├── kobest/          # KoBEST 5개 태스크
│   │   ├── haerae/          # HAE-RAE 6개 태스크
│   │   └── kmmlu/           # KMMLU 45개 분야
│   ├── evaluators/          # 평가 메트릭
│   ├── runners/             # 벤치마크 실행기
│   └── utils/               # 유틸리티
├── dashboard/               # Streamlit 대시보드
├── results/                 # 결과 저장
├── main.py                  # 메인 실행 스크립트
└── requirements.txt
```

## 참고 자료

- [KLUE Benchmark](https://klue-benchmark.com/)
- [KoBEST Paper](https://arxiv.org/abs/2204.04541)
- [HAE-RAE Bench](https://huggingface.co/datasets/HAERAE-HUB/HAE_RAE_BENCH)
- [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/)
- [Open Ko-LLM Leaderboard](https://huggingface.co/spaces/upstage/open-ko-llm-leaderboard)

## 변경 이력

### v1.1.0 (2025-12-26)

**새로운 기능**
- `--tasks` 옵션 추가: 특정 태스크만 선택적으로 실행 가능
- KMMLU 벤치마크 지원 추가 (45개 전문 분야)
- 대시보드 다국어 지원 개선 (영어/한국어)

**버그 수정**
- KLUE NER 태그 매핑 오류 수정 (TAG_MAP이 잘못되어 F1 점수가 0에 가까웠던 문제)
- NER 평가 시 문자 단위 토큰 처리 개선

### v1.0.0 (초기 릴리스)

- KLUE, KoBEST, HAE-RAE 벤치마크 지원
- AWS Bedrock Nova 모델 (Lite, Pro, Nova 2 Lite) 지원
- Streamlit 대시보드
- 자동 결과 저장 및 중복 실행 방지

## 라이선스

MIT License

## 기여

이슈 및 PR 환영합니다.
