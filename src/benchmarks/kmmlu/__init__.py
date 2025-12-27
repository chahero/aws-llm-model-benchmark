"""
KMMLU 벤치마크 (Korean MMLU)
"""
from src.benchmarks.kmmlu.kmmlu import KMMLUBenchmark


def get_all_kmmlu_benchmarks():
    """모든 KMMLU 벤치마크 반환"""
    return [KMMLUBenchmark()]


__all__ = ["KMMLUBenchmark", "get_all_kmmlu_benchmarks"]
