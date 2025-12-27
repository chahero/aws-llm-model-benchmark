# AWS Bedrock Korean LLM Benchmark

A benchmarking framework for evaluating Korean language performance of LLM models on AWS Bedrock.

**English** | [한국어](README.md)

## Overview

This project evaluates the Korean language processing capabilities of AWS Bedrock Foundation Models (Nova series) across various benchmarks.

### Key Features

- **4 Standard Benchmarks**: KLUE, KoBEST, HAE-RAE, KMMLU
- **27+ Tasks**: Inference, similarity, NER, reading comprehension, classification, etc.
- **Selective Task Execution**: Re-run specific tasks only (`--tasks` option)
- **Automatic Result Saving**: JSON and Markdown report generation
- **Web Dashboard**: Streamlit-based visualization
- **Cost Tracking**: Token usage and estimated cost calculation
- **Duplicate Prevention**: Automatically skips tests with identical conditions

## Supported Models

| Model | Model ID | Input (1K tokens) | Output (1K tokens) |
|-------|----------|-------------------|-------------------|
| nova-lite | us.amazon.nova-lite-v1:0 | $0.00006 | $0.00024 |
| nova-pro | us.amazon.nova-pro-v1:0 | $0.0008 | $0.0032 |
| nova-2-lite | us.amazon.nova-2-lite-v1:0 | $0.00033 | $0.00275 |
| claude-sonnet-4-5 | global.anthropic.claude-sonnet-4-5-20250929-v1:0 | $0.003 | $0.015 |

> Model IDs can be customized in the `.env` file.

## Benchmarks

### KLUE (Korean Language Understanding Evaluation)

| Task | Description | Metric |
|------|-------------|--------|
| NLI | Natural Language Inference | Accuracy |
| STS | Semantic Textual Similarity (0~5) | Pearson |
| NER | Named Entity Recognition | F1 |
| RE | Relation Extraction | F1 |
| MRC | Machine Reading Comprehension | F1/EM |
| DP | Dependency Parsing | UAS/LAS |
| WoS | Wizard of Seoul (Dialogue State) | JGA |
| YNAT | News Topic Classification | Accuracy |

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
| Loanwords | Foreign word notation | Accuracy |
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
python main.py --benchmarks klue

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
# Re-run NER task only (overwrite existing results)
python main.py --models all --benchmarks klue --tasks ner --force

# Run multiple selected tasks
python main.py --models all --benchmarks all --tasks ner boolq copa

# Run specific tasks for specific model
python main.py --models nova-lite --benchmarks klue --tasks ner mrc --force
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
BENCHMARK_TASKS=klue,kobest,haerae
BENCHMARK_SAMPLE_LIMIT=        # Leave empty for full dataset
BENCHMARK_RATE_LIMIT=10.0
```

### Run Dashboard

```bash
streamlit run dashboard/app.py
```

## Results

### Storage Location

```
results/
├── raw/                    # JSON result files
│   ├── nova-lite_klue_nli_n3000.json
│   └── ...
└── reports/                # Markdown reports
    ├── nova-lite_klue_nli_n3000.md
    └── summary.md
```

### Filename Format

```
{model}_{benchmark}_{task}_n{samples}.json
{model}_{benchmark}_{task}_full.json  # Full dataset
```

### Example Result

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
│   │   ├── klue/            # KLUE 8 tasks
│   │   ├── kobest/          # KoBEST 5 tasks
│   │   ├── haerae/          # HAE-RAE 6 tasks
│   │   └── kmmlu/           # KMMLU 45 domains
│   ├── evaluators/          # Evaluation metrics
│   ├── runners/             # Benchmark runners
│   └── utils/               # Utilities
├── dashboard/               # Streamlit dashboard
├── results/                 # Result storage
├── main.py                  # Main execution script
└── requirements.txt
```

## Benchmark Results (Nova Lite - Partial)

| Benchmark | Task | Samples | Metric | Score |
|-----------|------|---------|--------|-------|
| KLUE | NLI | 3,000 | Accuracy | 80.8% |
| KLUE | STS | 519 | Pearson | 0.875 |
| KLUE | YNAT | 9,107 | Accuracy | 68.0% |

*Full results will be updated upon completion.*

## Cost Estimation

| Model | ~50K API Calls | Time (Sequential) |
|-------|----------------|-------------------|
| Nova Lite | ~$1.50 | ~21 hours |
| Nova Pro | ~$20 | ~28 hours |
| Nova 2 Lite | ~$15 | ~24 hours |

**Total (3 models, full dataset): ~$35-40**

## References

- [KLUE Benchmark](https://klue-benchmark.com/)
- [KoBEST Paper](https://arxiv.org/abs/2204.04541)
- [HAE-RAE Bench](https://huggingface.co/datasets/HAERAE-HUB/HAE_RAE_BENCH)
- [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/)
- [Open Ko-LLM Leaderboard](https://huggingface.co/spaces/upstage/open-ko-llm-leaderboard)

## Changelog

### v1.1.0 (2025-12-26)

**New Features**
- Added `--tasks` option: Selectively run specific tasks only
- Added KMMLU benchmark support (45 professional domains)
- Improved dashboard multilingual support (English/Korean)

**Bug Fixes**
- Fixed KLUE NER tag mapping error (TAG_MAP was incorrect, causing F1 scores near 0)
- Improved character-level token handling in NER evaluation

### v1.0.0 (Initial Release)

- KLUE, KoBEST, HAE-RAE benchmark support
- AWS Bedrock Nova models (Lite, Pro, Nova 2 Lite) support
- Streamlit dashboard
- Automatic result saving and duplicate prevention

## License

MIT License

## Contributing

Issues and PRs are welcome.
