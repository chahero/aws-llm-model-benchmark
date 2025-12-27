"""
마크다운 리포트 생성기
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List


class ReportGenerator:
    """벤치마크 결과를 마크다운 리포트로 생성"""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.raw_dir = self.results_dir / "raw"
        self.reports_dir = self.results_dir / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def load_results(self) -> List[Dict[str, Any]]:
        """모든 결과 파일 로드"""
        results = []
        if self.raw_dir.exists():
            for file in sorted(self.raw_dir.glob("*.json")):
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    data["_filename"] = file.name
                    results.append(data)
        return results

    def generate_task_report(self, result: Dict[str, Any]) -> str:
        """단일 태스크 결과를 마크다운으로 변환"""
        model = result.get("model_name", "unknown")
        benchmark = result.get("benchmark_name", "unknown")
        task = result.get("task_name", "unknown")
        metrics = result.get("metrics", {})
        metadata = result.get("metadata", {})
        raw_outputs = result.get("raw_outputs", [])

        md = []
        md.append(f"# {benchmark.upper()} - {task.upper()}")
        md.append(f"\n**모델**: {model}")
        md.append(f"\n**실행 시간**: {metadata.get('timestamp', 'N/A')}")
        md.append("")

        # 메트릭
        md.append("## 평가 결과")
        md.append("")
        md.append("| 메트릭 | 값 |")
        md.append("|--------|-----|")
        for key, value in metrics.items():
            if isinstance(value, float):
                md.append(f"| {key} | {value:.4f} |")
            else:
                md.append(f"| {key} | {value} |")
        md.append("")

        # 메타데이터
        md.append("## 실행 정보")
        md.append("")
        md.append(f"- **예제 수**: {metadata.get('num_examples', 'N/A')}")
        md.append(f"- **유효 예측 수**: {metadata.get('num_valid', 'N/A')}")
        md.append(f"- **총 소요 시간**: {metadata.get('elapsed_time_sec', 0):.2f}초")
        md.append(f"- **평균 응답 시간**: {metadata.get('avg_latency_ms', 0):.2f}ms")
        md.append(f"- **예상 비용**: ${metadata.get('cost_estimate', 0):.6f}")
        md.append("")

        # 샘플 출력 (최대 3개)
        if raw_outputs:
            md.append("## 샘플 출력")
            md.append("")
            for i, output in enumerate(raw_outputs[:3]):
                md.append(f"### 예제 {i+1}")
                md.append("")
                md.append("**프롬프트:**")
                md.append("```")
                prompt = output.get("prompt", "")
                # 프롬프트가 너무 길면 자르기
                if len(prompt) > 500:
                    prompt = prompt[:500] + "..."
                md.append(prompt)
                md.append("```")
                md.append("")
                md.append("**모델 출력:**")
                md.append("```")
                md.append(output.get("model_output", "N/A"))
                md.append("```")
                md.append("")
                md.append(f"- **예측**: {output.get('prediction', 'N/A')}")
                md.append(f"- **정답**: {output.get('reference', 'N/A')}")
                md.append(f"- **정답 여부**: {'O' if output.get('prediction') == output.get('reference') else 'X'}")
                md.append("")

        return "\n".join(md)

    def generate_summary_report(self, results: List[Dict[str, Any]]) -> str:
        """전체 요약 리포트 생성"""
        md = []
        md.append("# AWS Bedrock 한국어 벤치마크 결과")
        md.append("")
        md.append(f"**생성 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        md.append("")

        # 모델별로 그룹화
        by_model: Dict[str, List[Dict]] = {}
        for result in results:
            model = result.get("model_name", "unknown")
            if model not in by_model:
                by_model[model] = []
            by_model[model].append(result)

        # 요약 테이블
        md.append("## 결과 요약")
        md.append("")

        for model, model_results in by_model.items():
            md.append(f"### {model}")
            md.append("")
            md.append("| 벤치마크 | 태스크 | 주요 메트릭 | 값 |")
            md.append("|----------|--------|------------|-----|")

            total_score = 0
            count = 0

            for result in model_results:
                benchmark = result.get("benchmark_name", "")
                task = result.get("task_name", "")
                metrics = result.get("metrics", {})

                # 주요 메트릭 선택
                main_metric = None
                main_value = None
                for key in ["accuracy", "f1", "exact_match", "pearson"]:
                    if key in metrics and isinstance(metrics[key], (int, float)):
                        main_metric = key
                        main_value = metrics[key]
                        break

                if main_metric and main_value is not None:
                    md.append(f"| {benchmark} | {task} | {main_metric} | {main_value:.4f} |")
                    total_score += main_value
                    count += 1

            if count > 0:
                avg_score = total_score / count
                md.append(f"| **평균** | | | **{avg_score:.4f}** |")
            md.append("")

        # 비용 요약
        md.append("## 비용 요약")
        md.append("")
        total_cost = sum(
            r.get("metadata", {}).get("cost_estimate", 0)
            for r in results
        )
        md.append(f"**총 예상 비용**: ${total_cost:.6f}")
        md.append("")

        return "\n".join(md)

    def save_task_report(self, result: Dict[str, Any]) -> Path:
        """단일 태스크 리포트 저장"""
        model = result.get("model_name", "unknown")
        benchmark = result.get("benchmark_name", "unknown")
        task = result.get("task_name", "unknown")
        num_examples = result.get("metadata", {}).get("num_examples", 0)

        # 조건 기반 파일명 (JSON과 동일한 형식)
        sample_str = f"n{num_examples}" if num_examples else "full"
        filename = f"{model}_{benchmark}_{task}_{sample_str}.md"
        filepath = self.reports_dir / filename

        report = self.generate_task_report(result)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report)

        return filepath

    def save_summary_report(self) -> Path:
        """전체 요약 리포트 저장"""
        results = self.load_results()
        if not results:
            return None

        # 단일 요약 파일 (덮어쓰기)
        filename = "summary.md"
        filepath = self.reports_dir / filename

        report = self.generate_summary_report(results)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report)

        return filepath

    def generate_all_reports(self) -> List[Path]:
        """모든 리포트 생성"""
        results = self.load_results()
        saved_files = set()

        # 개별 태스크 리포트 (중복 파일명은 자동으로 덮어쓰기)
        for result in results:
            filepath = self.save_task_report(result)
            saved_files.add(filepath)

        # 요약 리포트
        summary_path = self.save_summary_report()
        if summary_path:
            saved_files.add(summary_path)

        return list(saved_files)


def generate_reports():
    """리포트 생성 실행"""
    generator = ReportGenerator()
    files = generator.generate_all_reports()
    print(f"Generated {len(files)} report(s):")
    for f in files:
        print(f"  - {f}")


if __name__ == "__main__":
    generate_reports()
