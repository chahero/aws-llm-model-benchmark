"""
AWS Bedrock 한국어 벤치마크 메인 실행 스크립트
"""
import argparse
from typing import List

from config.settings import MODELS, benchmark_config
from src.runners.benchmark_runner import BatchBenchmarkRunner
from src.benchmarks.kobest import get_all_kobest_benchmarks
from src.benchmarks.haerae import get_all_haerae_benchmarks
from src.benchmarks.kmmlu import get_all_kmmlu_benchmarks
from src.utils.logger import get_logger
from src.utils.report_generator import ReportGenerator

logger = get_logger(__name__)


def get_benchmarks(benchmark_names: List[str], task_filter: List[str] = None):
    """
    벤치마크 인스턴스 반환

    Args:
        benchmark_names: 벤치마크 카테고리 (kobest, haerae, kmmlu, all)
        task_filter: 특정 태스크만 실행 (예: ["boolq", "copa"])
    """
    benchmarks = []

    if "kobest" in benchmark_names or "all" in benchmark_names:
        benchmarks.extend(get_all_kobest_benchmarks())


    if "haerae" in benchmark_names or "all" in benchmark_names:
        benchmarks.extend(get_all_haerae_benchmarks())

    if "kmmlu" in benchmark_names or "all" in benchmark_names:
        benchmarks.extend(get_all_kmmlu_benchmarks())

    # 특정 태스크 필터링
    if task_filter:
        benchmarks = [b for b in benchmarks if b.task_name in task_filter]

    return benchmarks


def main():
    # 환경변수에서 기본값 로드
    env_models = benchmark_config.selected_models
    env_benchmarks = benchmark_config.selected_benchmarks
    env_limit = benchmark_config.sample_limit
    env_rate_limit = benchmark_config.rate_limit

    parser = argparse.ArgumentParser(
        description="AWS Bedrock 한국어 모델 벤치마크",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 전체 벤치마크 실행
  python main.py --models all --benchmarks all --limit 100

  # 특정 태스크만 실행 (NER만 다시 테스트)
  python main.py --models all --benchmarks kobest --tasks boolq --force

  # 여러 태스크 선택 실행
  python main.py --models nova-pro --benchmarks all --tasks boolq copa copa

  # 특정 모델의 HAE-RAE 벤치마크만 실행
  python main.py --models nova-2-lite --benchmarks haerae
        """
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=f"테스트할 모델 (환경변수: {env_models})",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help=f"실행할 벤치마크 카테고리 (환경변수: {env_benchmarks})",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="실행할 특정 태스크 (예: --tasks boolq copa). 지정하면 해당 태스크만 실행",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=f"태스크당 최대 예제 수 (환경변수: {env_limit})",
    )
    parser.add_argument(
        "--output-dir",
        default="results/raw",
        help="결과 저장 디렉토리",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=None,
        help=f"초당 API 요청 수 제한 (환경변수: {env_rate_limit})",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="벤치마크 완료 후 마크다운 리포트 생성",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="벤치마크 없이 기존 결과로 리포트만 생성",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="기존 결과 무시하고 재실행",
    )

    args = parser.parse_args()

    # 리포트만 생성 모드
    if args.report_only:
        logger.info("리포트만 생성 모드")
        generator = ReportGenerator()
        files = generator.generate_all_reports()
        print(f"\n마크다운 리포트 {len(files)}개 생성 완료:")
        for f in files:
            print(f"  - {f}")
        return

    # CLI 인자 > 환경변수 > 기본값
    selected_models = args.models or env_models
    selected_benchmarks = args.benchmarks or env_benchmarks
    limit = args.limit if args.limit is not None else env_limit
    rate_limit = args.rate_limit if args.rate_limit is not None else env_rate_limit

    # 모델 목록
    if "all" in selected_models:
        model_names = list(MODELS.keys())
    else:
        model_names = selected_models

    # 벤치마크 목록 (태스크 필터 적용)
    benchmarks = get_benchmarks(selected_benchmarks, args.tasks)

    if not benchmarks:
        logger.error("No benchmarks selected")
        if args.tasks:
            logger.error(f"Task filter: {args.tasks} - 해당 태스크가 선택된 벤치마크에 없습니다.")
        return

    logger.info(f"Models: {model_names}")
    logger.info(f"Benchmarks: {[b.task_name for b in benchmarks]}")
    if args.tasks:
        logger.info(f"Task filter: {args.tasks}")
    logger.info(f"Limit: {limit}")

    # 배치 실행
    runner = BatchBenchmarkRunner(
        model_names=model_names,
        benchmarks=benchmarks,
        rate_limit=rate_limit,
    )

    results = runner.run_all(
        split="test",
        limit=limit,
        output_dir=args.output_dir,
        force=args.force,
    )

    # 결과 출력
    print("\n" + "=" * 60)
    print("벤치마크 결과 요약")
    print("=" * 60)

    for model_name, tasks in results.items():
        print(f"\n[{model_name}]")
        for task_name, result in tasks.items():
            metrics_str = ", ".join(
                f"{k}: {v:.4f}" for k, v in result.metrics.items()
                if isinstance(v, (int, float))
            )
            print(f"  {task_name}: {metrics_str}")

    # 마크다운 리포트 생성
    if args.report:
        print("\n" + "-" * 60)
        print("마크다운 리포트 생성 중...")
        generator = ReportGenerator()
        files = generator.generate_all_reports()
        print(f"리포트 {len(files)}개 생성 완료:")
        for f in files:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
