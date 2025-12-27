"""
벤치마크 실행기
"""
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from src.models.nova_models import NovaModel
from src.benchmarks.base_benchmark import BaseBenchmark, BenchmarkResult
from src.utils.logger import get_logger
from src.utils.rate_limiter import RateLimiter
from src.utils.cost_tracker import CostTracker
from config.settings import benchmark_config

logger = get_logger(__name__)


class BenchmarkRunner:
    """벤치마크 실행 오케스트레이터"""

    def __init__(
        self,
        model: NovaModel,
        rate_limit: float = None,
        retry_count: int = None,
        save_intermediate: bool = None,
    ):
        """
        Args:
            model: 테스트할 모델
            rate_limit: 초당 요청 수 제한
            retry_count: 재시도 횟수
            save_intermediate: 중간 결과 저장 여부
        """
        self.model = model
        self.rate_limiter = RateLimiter(
            rate_limit or benchmark_config.rate_limit
        )
        self.retry_count = retry_count or benchmark_config.retry_count
        self.save_intermediate = (
            save_intermediate
            if save_intermediate is not None
            else benchmark_config.save_intermediate
        )
        self.cost_tracker = CostTracker()

    def run_benchmark(
        self,
        benchmark: BaseBenchmark,
        split: str = "test",
        limit: Optional[int] = None,
        output_dir: str = "results/raw",
        force: bool = False,
    ) -> Optional[BenchmarkResult]:
        """
        단일 벤치마크 실행

        Args:
            benchmark: 실행할 벤치마크
            split: 데이터 분할
            limit: 최대 예제 수
            output_dir: 결과 저장 디렉토리
            force: 기존 결과 무시하고 재실행

        Returns:
            BenchmarkResult: 벤치마크 결과 (스킵시 None)
        """
        # 이미 결과가 존재하는지 확인
        filename = self._get_result_filename(
            self.model.model_name,
            benchmark.benchmark_name,
            benchmark.task_name,
            limit,
        )

        if not force and self._result_exists(output_dir, filename):
            logger.info(
                f"Skipping {benchmark.benchmark_name}/{benchmark.task_name} "
                f"on {self.model.model_name} (result exists: {filename})"
            )
            return None

        logger.info(
            f"Running {benchmark.benchmark_name}/{benchmark.task_name} "
            f"on {self.model.model_name}"
        )

        predictions = []
        references = []
        raw_outputs = []

        start_time = time.time()
        system_prompt = benchmark.get_system_prompt()

        # 예제 수집
        examples = list(benchmark.iterate_examples(split, limit))
        total_examples = len(examples)

        logger.info(f"Total examples: {total_examples}")

        # 진행 표시
        progress_bar = tqdm(
            examples,
            desc=f"{benchmark.task_name}",
            unit="ex",
        )

        for example in progress_bar:
            # Rate limiting
            self.rate_limiter.wait()

            prompt = benchmark.format_prompt(example)
            reference = benchmark.get_reference(example)

            # 모델 호출
            try:
                response = self._invoke_with_retry(prompt, system_prompt)
                prediction = benchmark.extract_answer(response.output)

                # 비용 추적
                self.cost_tracker.add_usage(
                    self.model.model_name,
                    response.input_tokens,
                    response.output_tokens,
                )

                raw_output = {
                    "prompt": prompt,
                    "model_output": response.output,
                    "prediction": prediction,
                    "reference": reference,
                    "latency_ms": response.latency_ms,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                }

            except Exception as e:
                logger.error(f"Error processing example: {e}")
                prediction = None
                raw_output = {
                    "prompt": prompt,
                    "error": str(e),
                    "prediction": None,
                    "reference": reference,
                }

            predictions.append(prediction)
            references.append(reference)
            raw_outputs.append(raw_output)

        elapsed_time = time.time() - start_time

        # 평가
        # None 값 필터링
        valid_pairs = [
            (p, r) for p, r in zip(predictions, references)
            if p is not None
        ]

        if valid_pairs:
            valid_preds, valid_refs = zip(*valid_pairs)
            metrics = benchmark.evaluate(list(valid_preds), list(valid_refs))
        else:
            metrics = {"error": "No valid predictions"}

        # 결과 생성
        result = BenchmarkResult(
            benchmark_name=benchmark.benchmark_name,
            task_name=benchmark.task_name,
            model_name=self.model.model_name,
            predictions=predictions,
            references=references,
            metrics=metrics,
            raw_outputs=raw_outputs,
            metadata={
                "split": split,
                "num_examples": total_examples,
                "num_valid": len(valid_pairs),
                "elapsed_time_sec": elapsed_time,
                "avg_latency_ms": sum(
                    o.get("latency_ms", 0) for o in raw_outputs
                ) / max(len(raw_outputs), 1),
                "timestamp": datetime.now().isoformat(),
                "cost_estimate": self.cost_tracker.get_total_cost(),
            },
        )

        # 결과 저장
        if self.save_intermediate:
            self._save_result(result, output_dir, limit)

        logger.info(f"Completed {benchmark.task_name}: {metrics}")
        return result

    def _invoke_with_retry(self, prompt: str, system_prompt: str):
        """재시도 로직 포함 모델 호출"""
        last_error = None

        for attempt in range(self.retry_count):
            try:
                return self.model.generate(prompt, system_prompt)
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")

                if attempt < self.retry_count - 1:
                    sleep_time = 2 ** attempt  # Exponential backoff
                    time.sleep(sleep_time)

        raise last_error

    def _get_result_filename(
        self,
        model_name: str,
        benchmark_name: str,
        task_name: str,
        num_examples: int,
    ) -> str:
        """결과 파일명 생성 (조건 기반)"""
        # 샘플 수가 None이면 'full'로 표시
        sample_str = f"n{num_examples}" if num_examples else "full"
        return f"{model_name}_{benchmark_name}_{task_name}_{sample_str}.json"

    def _result_exists(self, output_dir: str, filename: str) -> bool:
        """결과 파일 존재 여부 확인"""
        filepath = Path(output_dir) / filename
        return filepath.exists()

    def _save_result(self, result: BenchmarkResult, output_dir: str, num_examples: int = None):
        """결과 저장"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filename = self._get_result_filename(
            result.model_name,
            result.benchmark_name,
            result.task_name,
            num_examples or result.metadata.get("num_examples"),
        )
        filepath = output_path / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info(f"Result saved to {filepath}")


class BatchBenchmarkRunner:
    """여러 모델/벤치마크 배치 실행"""

    def __init__(
        self,
        model_names: List[str],
        benchmarks: List[BaseBenchmark],
        **runner_kwargs,
    ):
        """
        Args:
            model_names: 테스트할 모델 이름 리스트
            benchmarks: 실행할 벤치마크 리스트
            **runner_kwargs: BenchmarkRunner 추가 인자
        """
        self.model_names = model_names
        self.benchmarks = benchmarks
        self.runner_kwargs = runner_kwargs
        self.results: Dict[str, Dict[str, BenchmarkResult]] = {}

    def run_all(
        self,
        split: str = "test",
        limit: Optional[int] = None,
        output_dir: str = "results/raw",
        force: bool = False,
    ) -> Dict[str, Dict[str, BenchmarkResult]]:
        """
        모든 모델-벤치마크 조합 실행

        Args:
            split: 데이터 분할
            limit: 최대 예제 수
            output_dir: 결과 저장 디렉토리
            force: 기존 결과 무시하고 재실행

        Returns:
            Dict[model_name, Dict[task_name, BenchmarkResult]]
        """
        total_runs = len(self.model_names) * len(self.benchmarks)
        current_run = 0
        skipped_count = 0

        for model_name in self.model_names:
            logger.info(f"\n{'='*50}")
            logger.info(f"Testing model: {model_name}")
            logger.info(f"{'='*50}")

            model = NovaModel(model_name)
            runner = BenchmarkRunner(model, **self.runner_kwargs)
            self.results[model_name] = {}

            for benchmark in self.benchmarks:
                current_run += 1
                logger.info(
                    f"\n[{current_run}/{total_runs}] "
                    f"{model_name} - {benchmark.task_name}"
                )

                try:
                    result = runner.run_benchmark(
                        benchmark, split, limit, output_dir, force
                    )
                    if result is None:
                        skipped_count += 1
                    else:
                        self.results[model_name][benchmark.task_name] = result
                except Exception as e:
                    logger.error(f"Failed: {e}")
                    continue

        if skipped_count > 0:
            logger.info(f"\nSkipped {skipped_count} benchmark(s) (already completed)")

        return self.results

    def get_summary(self) -> Dict[str, Any]:
        """결과 요약 반환"""
        summary = {
            "models": list(self.results.keys()),
            "benchmarks": [b.task_name for b in self.benchmarks],
            "results": {},
        }

        for model_name, tasks in self.results.items():
            summary["results"][model_name] = {}
            for task_name, result in tasks.items():
                summary["results"][model_name][task_name] = result.metrics

        return summary

    def save_summary(self, output_path: str = "results/summary.json"):
        """요약 결과 저장"""
        summary = self.get_summary()
        summary["timestamp"] = datetime.now().isoformat()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"Summary saved to {output_path}")
