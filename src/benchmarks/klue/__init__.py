"""
KLUE 벤치마크 모듈
"""
from src.benchmarks.klue.nli import NLIBenchmark
from src.benchmarks.klue.sts import STSBenchmark
from src.benchmarks.klue.ynat import YNATBenchmark
from src.benchmarks.klue.mrc import MRCBenchmark
from src.benchmarks.klue.ner import NERBenchmark
from src.benchmarks.klue.re import REBenchmark
from src.benchmarks.klue.dp import DPBenchmark
from src.benchmarks.klue.wos import WOSBenchmark

__all__ = [
    "NLIBenchmark",
    "STSBenchmark",
    "YNATBenchmark",
    "MRCBenchmark",
    "NERBenchmark",
    "REBenchmark",
    "DPBenchmark",
    "WOSBenchmark",
]


def get_all_klue_benchmarks():
    """모든 KLUE 벤치마크 반환"""
    return [
        NLIBenchmark(),
        STSBenchmark(),
        YNATBenchmark(),
        MRCBenchmark(),
        NERBenchmark(),
        REBenchmark(),
        DPBenchmark(),
        WOSBenchmark(),
    ]
