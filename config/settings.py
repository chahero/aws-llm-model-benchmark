"""
AWS Bedrock 벤치마크 설정
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent

# HuggingFace 데이터셋 캐시 경로 설정
_hf_cache = os.getenv("HF_DATASETS_CACHE", "./data/datasets")
if not os.path.isabs(_hf_cache):
    _hf_cache = str(PROJECT_ROOT / _hf_cache)
os.environ["HF_DATASETS_CACHE"] = _hf_cache
os.environ["HF_HOME"] = str(PROJECT_ROOT / "data" / "huggingface")

# 캐시 디렉토리 생성
Path(_hf_cache).mkdir(parents=True, exist_ok=True)


@dataclass
class AWSConfig:
    """AWS 설정"""
    region: str = os.getenv("AWS_REGION", "us-east-1")
    access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    secret_access_key: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")


@dataclass
class ModelConfig:
    """모델 설정"""
    # 기본 추론 파라미터
    DEFAULT_MAX_TOKENS: int = 1024
    DEFAULT_TEMPERATURE: float = 0.0  # 재현성을 위해 0
    DEFAULT_TOP_P: float = 1.0


# 모델 이름 -> ID 매핑 (환경변수에서 읽거나 기본값 사용)
MODELS: Dict[str, str] = {
    "nova-lite": os.getenv("MODEL_ID_NOVA_LITE", "us.amazon.nova-lite-v1:0"),
    "nova-pro": os.getenv("MODEL_ID_NOVA_PRO", "us.amazon.nova-pro-v1:0"),
    "nova-2-lite": os.getenv("MODEL_ID_NOVA_2_LITE", "us.amazon.nova-2-lite-v1:0"),
    "claude-sonnet-4-5": os.getenv("MODEL_ID_CLAUDE_SONNET_4_5", "global.anthropic.claude-sonnet-4-5-20250929-v1:0"),
}

# 환경변수로 비활성화된 모델 제거
_disabled_models = os.getenv("DISABLED_MODELS", "").split(",")
MODELS = {k: v for k, v in MODELS.items() if k not in _disabled_models and v}

# 모델별 비용 (1K 토큰당 USD) - AWS 공식 가격
# https://aws.amazon.com/bedrock/pricing/
MODEL_COSTS: Dict[str, Dict[str, float]] = {
    "nova-lite": {"input": 0.00006, "output": 0.00024},
    "nova-pro": {"input": 0.0008, "output": 0.0032},
    "nova-2-lite": {"input": 0.00033, "output": 0.00275},
    "claude-sonnet-4-5": {"input": 0.003, "output": 0.015},
}


def _parse_list(env_var: str, default: list) -> list:
    """환경변수를 리스트로 파싱"""
    value = os.getenv(env_var, "")
    if not value:
        return default
    if value.lower() == "all":
        return ["all"]
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_int(env_var: str, default: int = None) -> int:
    """환경변수를 정수로 파싱"""
    value = os.getenv(env_var, "")
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _parse_float(env_var: str, default: float) -> float:
    """환경변수를 실수로 파싱"""
    value = os.getenv(env_var, "")
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


@dataclass
class BenchmarkConfig:
    """벤치마크 실행 설정"""
    # 데이터셋 설정
    default_split: str = "test"
    sample_limit: int = field(
        default_factory=lambda: _parse_int("BENCHMARK_SAMPLE_LIMIT", None)
    )

    # API 호출 설정
    rate_limit: float = field(
        default_factory=lambda: _parse_float("BENCHMARK_RATE_LIMIT", 10.0)
    )
    retry_count: int = 3
    retry_delay: float = 1.0  # seconds

    # 저장 설정
    save_intermediate: bool = True
    results_dir: str = "results"

    # 환경변수에서 읽은 설정
    selected_models: list = field(
        default_factory=lambda: _parse_list("BENCHMARK_MODELS", ["nova-micro"])
    )
    selected_benchmarks: list = field(
        default_factory=lambda: _parse_list("BENCHMARK_TASKS", ["kobest"])
    )


@dataclass
class DatasetConfig:
    """데이터셋 경로 설정"""
    # HuggingFace 데이터셋 경로
    KOBEST: str = "skt/kobest_v1"
    HAERAE: str = "HAERAE-HUB/HAE_RAE_BENCH"


# 벤치마크별 태스크 목록
BENCHMARK_TASKS: Dict[str, list] = {
    "kobest": ["boolq", "copa", "wic", "hellaswag", "sentineg"],
    "haerae": [
        "general_knowledge", "history", "reading_comprehension",
        "standard_nomenclature", "loan_words", "rare_words"
    ],
    "kmmlu": ["all"],
}


# 전역 설정 인스턴스
aws_config = AWSConfig()
model_config = ModelConfig()
benchmark_config = BenchmarkConfig()
dataset_config = DatasetConfig()
