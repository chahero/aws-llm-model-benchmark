# AWS Bedrock Korean LLM Benchmark

AWS Bedrock 기반 LLM 모델의 한국어 성능을 평가하는 벤치마크 프레임워크입니다.

[English](README_EN.md) | **한국어**

## 벤치마크 결과

> AWS Bedrock에서 제공하는 모델들의 한국어 벤치마크 결과입니다.

### 전체 결과 요약

> Claude Sonnet 4.5는 100개 샘플로 테스트, Nova 시리즈는 전체 데이터셋으로 테스트되었습니다.

| 벤치마크 | 태스크 | Claude Sonnet 4.5 | Nova Pro | Nova Lite | Nova 2 Lite |
|----------|--------|:-----------------:|:--------:|:---------:|:-----------:|
| **KoBEST** | BoolQ | **97.0%** | 97.1% | 95.6% | 81.6% |
| | COPA | **100.0%** | 95.5% | 93.0% | 92.6% |
| | HellaSwag | **93.0%** | 70.4% | 65.8% | 65.6% |
| | SentiNeg | **99.0%** | 96.5% | 96.7% | 94.7% |
| | WiC | **89.0%** | 79.2% | 80.2% | 74.0% |
| **HAE-RAE** | General Knowledge | **80.0%** | 58.0% | 42.6% | 44.3% |
| | History | **98.0%** | 82.4% | 72.3% | 63.3% |
| | Loan Words | **95.0%** | 74.0% | 63.9% | 62.7% |
| | Rare Words | **87.0%** | 77.0% | 56.5% | 55.8% |
| | Reading Comprehension | **77.0%** | 73.0% | 63.7% | 59.6% |
| | Standard Nomenclature | **96.0%** | 79.1% | 70.6% | 58.2% |
| **KMMLU** | All (45 domains) | **94.0%** | 55.7% | 40.3% | 55.3% |
| | | | | | |
| **평균** | | **92.1%** | **78.1%** | **70.1%** | **67.3%** |

### 모델별 순위

1. **Claude Sonnet 4.5** - 92.1% (모든 태스크에서 최고 성능)
2. **Nova Pro** - 78.1%
3. **Nova Lite** - 70.1%
4. **Nova 2 Lite** - 67.3%

## 개요

이 프로젝트는 AWS Bedrock에서 제공하는 Foundation Model(Nova 시리즈)의 한국어 처리 능력을 다양한 벤치마크로 평가합니다.

### 주요 기능

- **3개 표준 벤치마크 지원**: KoBEST, HAE-RAE, KMMLU
- **12개 태스크**: 추론, 독해, 상식, 어휘, 지식 평가 등
- **선택적 태스크 실행**: 특정 태스크만 재실행 가능 (`--tasks` 옵션)
- **자동 결과 저장**: JSON 및 마크다운 리포트 생성
- **비용 추적**: 토큰 사용량 및 예상 비용 계산
- **중복 실행 방지**: 동일 조건 테스트 자동 스킵

## 지원 모델

| 모델 | Model ID | 입력 (1K 토큰) | 출력 (1K 토큰) |
|------|----------|---------------|---------------|
| Nova Lite | us.amazon.nova-lite-v1:0 | $0.00006 | $0.00024 |
| Nova Pro | us.amazon.nova-pro-v1:0 | $0.0008 | $0.0032 |
| Nova 2 Lite | us.amazon.nova-2-lite-v1:0 | $0.00033 | $0.00275 |
| Claude Sonnet 4.5 | global.anthropic.claude-sonnet-4-5-20250929-v1:0 | $0.003 | $0.015 |

> Model ID는 `.env` 파일에서 커스터마이즈 가능합니다.

## 벤치마크

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
| Standard Nomenclature | 올바른 표현 선택 | Accuracy |
| Loan Words | 외래어 표기법 | Accuracy |
| Rare Words | 어휘 이해 | Accuracy |
| General Knowledge | 한국 상식 | Accuracy |
| History | 한국사 지식 | Accuracy |
| Reading Comprehension | 지문 이해 | Accuracy |

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
python main.py --benchmarks kobest

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
# BoolQ 태스크만 다시 실행 (기존 결과 덮어쓰기)
python main.py --models all --benchmarks kobest --tasks boolq --force

# 여러 태스크 선택 실행
python main.py --models all --benchmarks all --tasks boolq copa

# 특정 모델의 특정 태스크만 실행
python main.py --models nova-lite --benchmarks kobest --tasks boolq wic --force
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
BENCHMARK_TASKS=kobest,haerae,kmmlu
BENCHMARK_SAMPLE_LIMIT=        # 비워두면 전체 데이터
BENCHMARK_RATE_LIMIT=10.0
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
│   │   ├── kobest/          # KoBEST 5개 태스크
│   │   ├── haerae/          # HAE-RAE 6개 태스크
│   │   └── kmmlu/           # KMMLU 45개 분야
│   ├── evaluators/          # 평가 메트릭
│   ├── runners/             # 벤치마크 실행기
│   └── utils/               # 유틸리티
├── results/                 # 결과 저장
├── main.py                  # 메인 실행 스크립트
└── requirements.txt
```

## 참고 자료

- [KoBEST Paper](https://arxiv.org/abs/2204.04541)
- [HAE-RAE Bench](https://huggingface.co/datasets/HAERAE-HUB/HAE_RAE_BENCH)
- [KMMLU](https://huggingface.co/datasets/HAERAE-HUB/KMMLU)
- [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/)
- [Open Ko-LLM Leaderboard](https://huggingface.co/spaces/upstage/open-ko-llm-leaderboard)

## 변경 이력

### v1.2.0 (2024-12-28)

**변경 사항**
- KLUE 벤치마크 제거 (LLM 평가에 적합하지 않은 DP, WOS 등 태스크 포함)
- KoBEST, HAE-RAE, KMMLU 3개 벤치마크로 집중
- 이진 분류 태스크 답변 추출 로직 개선

### v1.1.0 (2024-12-26)

**새로운 기능**
- `--tasks` 옵션 추가: 특정 태스크만 선택적으로 실행 가능
- KMMLU 벤치마크 지원 추가 (45개 전문 분야)
- 대시보드 다국어 지원 개선 (영어/한국어)

### v1.0.0 (초기 릴리스)

- KoBEST, HAE-RAE 벤치마크 지원
- AWS Bedrock Nova 모델 (Lite, Pro, Nova 2 Lite) 지원
- 자동 결과 저장 및 중복 실행 방지

## 라이선스

MIT License

## 기여

이슈 및 PR 환영합니다.
