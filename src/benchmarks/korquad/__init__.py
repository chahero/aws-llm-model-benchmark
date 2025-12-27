"""
KorQuAD 벤치마크
"""
from src.benchmarks.korquad.korquad import KorQuADBenchmark


def get_all_korquad_benchmarks():
    """모든 KorQuAD 벤치마크 반환"""
    return [KorQuADBenchmark()]


__all__ = ["KorQuADBenchmark", "get_all_korquad_benchmarks"]
