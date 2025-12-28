# AWS Bedrock Korean LLM Benchmark

A benchmarking framework for evaluating Korean language performance of LLM models on AWS Bedrock.

**English** | [한국어](README.md)

## Benchmark Results

> December 2024 benchmark results for models available on AWS Bedrock.

### Summary

| Benchmark | Task | Claude Sonnet 4.5 | Nova Pro | Nova Lite | Nova 2 Lite |
|-----------|------|:-----------------:|:--------:|:---------:|:-----------:|
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
| **Average** | | **92.1%** | **78.1%** | **70.1%** | **67.3%** |

### Model Rankings

1. **Claude Sonnet 4.5** - 92.1% (Best performance across all tasks)
2. **Nova Pro** - 78.1%
3. **Nova Lite** - 70.1%
4. **Nova 2 Lite** - 67.3%

## Overview

This project evaluates the Korean language processing capabilities of AWS Bedrock Foundation Models (Nova series) across various benchmarks.

### Key Features

- **3 Standard Benchmarks**: KoBEST, HAE-RAE, KMMLU
- **12 Tasks**: Reasoning, reading comprehension, common sense, vocabulary, knowledge evaluation
- **Selective Task Execution**: Re-run specific tasks only (`--tasks` option)
- **Automatic Result Saving**: JSON and Markdown report generation
- **Cost Tracking**: Token usage and estimated cost calculation
- **Duplicate Prevention**: Automatically skips tests with identical conditions

## Supported Models

| Model | Model ID | Input (1K tokens) | Output (1K tokens) |
|-------|----------|-------------------|-------------------|
| Nova Lite | us.amazon.nova-lite-v1:0 | $0.00006 | $0.00024 |
| Nova Pro | us.amazon.nova-pro-v1:0 | $0.0008 | $0.0032 |
| Nova 2 Lite | us.amazon.nova-2-lite-v1:0 | $0.00033 | $0.00275 |
| Claude Sonnet 4.5 | global.anthropic.claude-sonnet-4-5-20250929-v1:0 | $0.003 | $0.015 |

> Model IDs can be customized in the `.env` file.

## Benchmarks

### KoBEST (Korean Balanced Evaluation of Significant Tasks)

| Task | Description | Metric |
|------|-------------|--------|
| BoolQ | Yes/No Questions | Accuracy |
| COPA | Causal Reasoning | Accuracy |
| WiC | Word-in-Context | Accuracy |
| HellaSwag | Commonsense Reasoning | Accuracy |
| SentiNeg | Sentiment Negation | Accuracy |

### HAE-RAE (Korean Cultural & Linguistic Benchmark)

| Task | Description | Metric |
|------|-------------|--------|
| Standard Nomenclature | Correct Korean expressions | Accuracy |
| Loan Words | Foreign word notation | Accuracy |
| Rare Words | Vocabulary understanding | Accuracy |
| General Knowledge | Korean common sense | Accuracy |
| History | Korean history knowledge | Accuracy |
| Reading Comprehension | Passage understanding | Accuracy |

### KMMLU (Korean Massive Multitask Language Understanding)

| Task | Description | Metric |
|------|-------------|--------|
| KMMLU | 45-domain professional knowledge evaluation | Accuracy |

*Includes 45 professional domains: Accounting, Law, Medicine, Engineering, History, Mathematics, Physics, etc.*

## Installation

### Requirements

- Python 3.9+
- AWS account with Bedrock access

### Setup

```bash
# Clone repository
git clone https://github.com/your-repo/aws-llm-model-benchmark.git
cd aws-llm-model-benchmark

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env file with your AWS credentials
```

## Usage

### Basic Execution

```bash
# Run full benchmark (all models, all tasks)
python main.py

# Test specific model only
python main.py --models nova-lite

# Test specific benchmark only
python main.py --benchmarks kobest

# Limit samples (for quick testing)
python main.py --limit 100

# Force re-run (ignore existing results)
python main.py --force

# Generate markdown reports
python main.py --report
```

### Selective Task Execution

Run specific tasks only to save time:

```bash
# Re-run BoolQ task only (overwrite existing results)
python main.py --models all --benchmarks kobest --tasks boolq --force

# Run multiple selected tasks
python main.py --models all --benchmarks all --tasks boolq copa

# Run specific tasks for specific model
python main.py --models nova-lite --benchmarks kobest --tasks boolq wic --force
```

### Environment Variables (.env)

```bash
# AWS Configuration
AWS_REGION=us-west-2
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret

# HuggingFace Token (for dataset download)
HF_TOKEN=your_hf_token

# Benchmark Configuration
BENCHMARK_MODELS=nova-lite,nova-pro,nova-2-lite
BENCHMARK_TASKS=kobest,haerae,kmmlu
BENCHMARK_SAMPLE_LIMIT=        # Leave empty for full dataset
BENCHMARK_RATE_LIMIT=10.0
```

## Project Structure

```
aws-llm-model-benchmark/
├── config/
│   └── settings.py          # Settings (model IDs, costs, etc.)
├── src/
│   ├── models/
│   │   ├── bedrock_client.py    # AWS Bedrock API
│   │   └── nova_models.py       # Nova model wrapper
│   ├── benchmarks/
│   │   ├── kobest/          # KoBEST 5 tasks
│   │   ├── haerae/          # HAE-RAE 6 tasks
│   │   └── kmmlu/           # KMMLU 45 domains
│   ├── evaluators/          # Evaluation metrics
│   ├── runners/             # Benchmark runners
│   └── utils/               # Utilities
├── results/                 # Result storage
├── main.py                  # Main execution script
└── requirements.txt
```

## References

- [KoBEST Paper](https://arxiv.org/abs/2204.04541)
- [HAE-RAE Bench](https://huggingface.co/datasets/HAERAE-HUB/HAE_RAE_BENCH)
- [KMMLU](https://huggingface.co/datasets/HAERAE-HUB/KMMLU)
- [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/)
- [Open Ko-LLM Leaderboard](https://huggingface.co/spaces/upstage/open-ko-llm-leaderboard)

## Changelog

### v1.2.0 (2024-12-28)

**Changes**
- Removed KLUE benchmark (contains tasks not suitable for LLM evaluation like DP, WOS)
- Focus on 3 benchmarks: KoBEST, HAE-RAE, KMMLU
- Improved binary classification answer extraction logic

### v1.1.0 (2024-12-26)

**New Features**
- Added `--tasks` option: Selectively run specific tasks only
- Added KMMLU benchmark support (45 professional domains)

### v1.0.0 (Initial Release)

- KoBEST, HAE-RAE benchmark support
- AWS Bedrock Nova models (Lite, Pro, Nova 2 Lite) support
- Automatic result saving and duplicate prevention

## License

MIT License

## Contributing

Issues and PRs are welcome.
