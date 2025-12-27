"""
KoBEST 벤치마크 모듈
"""
from src.benchmarks.kobest.boolq import BoolQBenchmark
from src.benchmarks.kobest.copa import COPABenchmark
from src.benchmarks.kobest.wic import WiCBenchmark
from src.benchmarks.kobest.hellaswag import HellaSwagBenchmark
from src.benchmarks.kobest.sentineg import SentiNegBenchmark

__all__ = [
    "BoolQBenchmark",
    "COPABenchmark",
    "WiCBenchmark",
    "HellaSwagBenchmark",
    "SentiNegBenchmark",
]


def get_all_kobest_benchmarks():
    """모든 KoBEST 벤치마크 반환"""
    return [
        BoolQBenchmark(),
        COPABenchmark(),
        WiCBenchmark(),
        HellaSwagBenchmark(),
        SentiNegBenchmark(),
    ]
