"""
AWS Bedrock Korean Benchmark Dashboard
"""
import json
from pathlib import Path
from typing import Dict, Any, List

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# 페이지 설정
st.set_page_config(
    page_title="AWS Bedrock Korean Benchmark",
    page_icon="🇰🇷",
    layout="wide",
)

# 다국어 지원
TRANSLATIONS = {
    "ko": {
        "title": "🇰🇷 AWS Bedrock 한국어 벤치마크 대시보드",
        "no_results": "벤치마크 결과가 없습니다. 먼저 벤치마크를 실행하세요.",
        "filter": "📌 필터",
        "select_model": "모델 선택",
        "benchmark_category": "벤치마크 카테고리",
        "metric_desc": "📖 메트릭 설명",
        "tab_comparison": "📊 전체 비교",
        "tab_leaderboard": "🏆 리더보드",
        "tab_visualization": "📈 시각화",
        "tab_details": "📋 상세 결과",
        "tab_info": "ℹ️ 벤치마크 설명",
        "comparison_header": "📊 전체 모델 비교",
        "comparison_desc": "모든 태스크에서 각 모델의 주요 지표를 한눈에 비교합니다.",
        "best_score": "🏆 최고 점수",
        "worst_score": "최저 점수",
        "score_range": "점수 범위: 0~1 (높을수록 좋음)",
        "benchmark_legend": "벤치마크 범례:",
        "download_csv": "📥 CSV 다운로드",
        "no_data": "비교할 데이터가 없습니다.",
        "task_comparison": "📊 태스크별 모델 성능 비교",
        "leaderboard_header": "🏆 리더보드",
        "num_models": "테스트된 모델 수",
        "num_tasks": "벤치마크 태스크 수",
        "total_evals": "총 평가 수",
        "most_wins": "🥇 가장 많이 1위",
        "model_wins": "🎯 모델별 1위 태스크 수",
        "overall_ranking": "📋 종합 순위",
        "ranking_desc": "각 모델의 모든 태스크 평균 점수 기준",
        "category_performance": "📁 벤치마크 카테고리별 성능",
        "radar_chart": "🎯 레이더 차트",
        "radar_desc": "각 모델의 벤치마크별 성능을 한눈에 비교",
        "heatmap": "🗺️ 히트맵",
        "heatmap_desc": "녹색일수록 높은 점수, 빨간색일수록 낮은 점수",
        "benchmark_detail": "📊 벤치마크별 상세 비교",
        "select_benchmark": "벤치마크 선택",
        "details_header": "📋 상세 결과",
        "info_header": "ℹ️ 벤치마크 설명",
        "benchmark_categories": "벤치마크 카테고리",
        "benchmark_intro": "본 대시보드는 다음 4개의 한국어 벤치마크를 사용합니다:",
        "metric_explanation": "📏 평가 메트릭 설명",
        "col_benchmark": "벤치마크",
        "col_task": "태스크",
        "col_desc": "설명",
        "col_model": "모델",
        "col_rank": "순위",
        "col_avg_score": "평균 점수",
        "col_max_score": "최고 점수",
        "col_min_score": "최저 점수",
        "col_num_tasks": "태스크 수",
        "col_wins": "1위 횟수",
        "col_ratio": "비율",
        "col_category": "카테고리",
        "col_metric": "메트릭",
        "col_score": "점수",
    },
    "en": {
        "title": "🇰🇷 AWS Bedrock Korean Benchmark Dashboard",
        "no_results": "No benchmark results found. Please run benchmarks first.",
        "filter": "📌 Filter",
        "select_model": "Select Model",
        "benchmark_category": "Benchmark Category",
        "metric_desc": "📖 Metric Description",
        "tab_comparison": "📊 Comparison",
        "tab_leaderboard": "🏆 Leaderboard",
        "tab_visualization": "📈 Visualization",
        "tab_details": "📋 Details",
        "tab_info": "ℹ️ About Benchmarks",
        "comparison_header": "📊 Model Comparison",
        "comparison_desc": "Compare all models across all tasks at a glance.",
        "best_score": "🏆 Best Score",
        "worst_score": "Worst Score",
        "score_range": "Score range: 0~1 (higher is better)",
        "benchmark_legend": "Benchmark Legend:",
        "download_csv": "📥 Download CSV",
        "no_data": "No data to compare.",
        "task_comparison": "📊 Model Performance by Task",
        "leaderboard_header": "🏆 Leaderboard",
        "num_models": "Number of Models",
        "num_tasks": "Number of Tasks",
        "total_evals": "Total Evaluations",
        "most_wins": "🥇 Most Wins",
        "model_wins": "🎯 Wins per Model",
        "overall_ranking": "📋 Overall Ranking",
        "ranking_desc": "Based on average score across all tasks",
        "category_performance": "📁 Performance by Category",
        "radar_chart": "🎯 Radar Chart",
        "radar_desc": "Compare model performance across benchmarks",
        "heatmap": "🗺️ Heatmap",
        "heatmap_desc": "Green = high score, Red = low score",
        "benchmark_detail": "📊 Detailed Benchmark Comparison",
        "select_benchmark": "Select Benchmark",
        "details_header": "📋 Detailed Results",
        "info_header": "ℹ️ About Benchmarks",
        "benchmark_categories": "Benchmark Categories",
        "benchmark_intro": "This dashboard uses the following 4 Korean benchmarks:",
        "metric_explanation": "📏 Metric Explanation",
        "col_benchmark": "Benchmark",
        "col_task": "Task",
        "col_desc": "Description",
        "col_model": "Model",
        "col_rank": "Rank",
        "col_avg_score": "Avg Score",
        "col_max_score": "Max Score",
        "col_min_score": "Min Score",
        "col_num_tasks": "# Tasks",
        "col_wins": "Wins",
        "col_ratio": "Ratio",
        "col_category": "Category",
        "col_metric": "Metric",
        "col_score": "Score",
    }
}

def get_text(key: str, lang: str = "ko") -> str:
    """Get translated text"""
    return TRANSLATIONS.get(lang, TRANSLATIONS["ko"]).get(key, key)


# 컬럼명 번역 매핑
COLUMN_TRANSLATIONS = {
    "ko": {
        "벤치마크": "벤치마크",
        "태스크": "태스크",
        "설명": "설명",
        "순위": "순위",
        "모델": "모델",
        "평균 점수": "평균 점수",
        "최고 점수": "최고 점수",
        "최저 점수": "최저 점수",
        "태스크 수": "태스크 수",
        "카테고리": "카테고리",
        "점수": "점수",
    },
    "en": {
        "벤치마크": "Benchmark",
        "태스크": "Task",
        "설명": "Description",
        "순위": "Rank",
        "모델": "Model",
        "평균 점수": "Avg Score",
        "최고 점수": "Max Score",
        "최저 점수": "Min Score",
        "태스크 수": "# Tasks",
        "카테고리": "Category",
        "점수": "Score",
    }
}


def translate_columns(df: pd.DataFrame, lang: str) -> pd.DataFrame:
    """DataFrame 컬럼명 번역"""
    translations = COLUMN_TRANSLATIONS.get(lang, COLUMN_TRANSLATIONS["ko"])
    return df.rename(columns=translations)

# 태스크 -> 벤치마크 매핑
TASK_TO_BENCHMARK = {
    # KoBEST
    "boolq": "KoBEST",
    "copa": "KoBEST",
    "wic": "KoBEST",
    "hellaswag": "KoBEST",
    "sentineg": "KoBEST",
    # HAE-RAE
    "general_knowledge": "HAE-RAE",
    "history": "HAE-RAE",
    "reading_comprehension": "HAE-RAE",
    "standard_nomenclature": "HAE-RAE",
    "loan_words": "HAE-RAE",
    "rare_words": "HAE-RAE",
    # KMMLU
    "kmmlu": "KMMLU",
}

# 벤치마크 색상
BENCHMARK_COLORS = {
    "KoBEST": "#3498db",   # 파랑
    "HAE-RAE": "#27ae60",  # 초록
    "KMMLU": "#9b59b6",    # 보라
}

# 메트릭 설명 정의 (다국어)
METRIC_DESCRIPTIONS = {
    "ko": {
        "accuracy": "정확도 - 전체 예측 중 정답 비율 (0~1, 높을수록 좋음)",
        "f1": "F1 점수 - 정밀도와 재현율의 조화평균 (0~1, 높을수록 좋음)",
        "pearson": "피어슨 상관계수 - 예측값과 실제값의 선형 상관관계 (-1~1, 1에 가까울수록 좋음)",
        "exact_match": "완전 일치 - 예측이 정답과 정확히 일치하는 비율 (0~1)",
        "precision": "정밀도 - 예측한 것 중 실제 정답 비율 (0~1)",
        "recall": "재현율 - 실제 정답 중 예측한 비율 (0~1)",
        "micro_f1": "Micro F1 - 전체 샘플 기준 F1 점수",
        "macro_f1": "Macro F1 - 클래스별 F1의 평균",
        "joint_goal_accuracy": "Joint Goal Accuracy - 대화 상태 추적 정확도",
    },
    "en": {
        "accuracy": "Accuracy - Ratio of correct predictions (0~1, higher is better)",
        "f1": "F1 Score - Harmonic mean of precision and recall (0~1, higher is better)",
        "pearson": "Pearson - Linear correlation between prediction and actual (-1~1, closer to 1 is better)",
        "exact_match": "Exact Match - Ratio of exactly matching predictions (0~1)",
        "precision": "Precision - Ratio of true positives among predictions (0~1)",
        "recall": "Recall - Ratio of true positives among actual positives (0~1)",
        "micro_f1": "Micro F1 - F1 score based on total samples",
        "macro_f1": "Macro F1 - Average of class-wise F1 scores",
        "joint_goal_accuracy": "Joint Goal Accuracy - Dialog state tracking accuracy",
    }
}

# 벤치마크/태스크 설명 정의 (다국어)
TASK_DESCRIPTIONS = {
    "ko": {
        # KoBEST
        "boolq": "BoolQ - 예/아니오 질의응답 (이진 분류)",
        "copa": "COPA - 인과 추론 (원인-결과 관계 파악)",
        "wic": "WiC - 문맥 속 단어 의미 구별",
        "hellaswag": "HellaSwag - 상식 기반 문장 완성",
        "sentineg": "SentiNeg - 부정 표현 감정 분석",
        # HAE-RAE
        "general_knowledge": "일반 상식 - 한국 문화/상식 지식",
        "history": "한국사 - 역사적 사실 및 사건",
        "reading_comprehension": "독해 - 지문 이해 및 추론",
        "standard_nomenclature": "표준어/맞춤법 - 올바른 표기법",
        "loan_words": "외래어 - 외래어 표기법",
        "rare_words": "희귀 단어 - 어휘력 평가",
        # KMMLU
        "kmmlu": "KMMLU - 한국어 다분야 지식 평가 (45개 분야)",
    },
    "en": {
        # KoBEST
        "boolq": "BoolQ - Yes/No Question Answering (Binary Classification)",
        "copa": "COPA - Causal Reasoning (Cause-Effect Relationship)",
        "wic": "WiC - Word Sense Disambiguation in Context",
        "hellaswag": "HellaSwag - Commonsense Sentence Completion",
        "sentineg": "SentiNeg - Negation Sentiment Analysis",
        # HAE-RAE
        "general_knowledge": "General Knowledge - Korean Culture & Common Sense",
        "history": "History - Korean Historical Facts & Events",
        "reading_comprehension": "Reading Comprehension - Text Understanding",
        "standard_nomenclature": "Standard Nomenclature - Correct Spelling/Grammar",
        "loan_words": "Loan Words - Foreign Word Notation",
        "rare_words": "Rare Words - Vocabulary Assessment",
        # KMMLU
        "kmmlu": "KMMLU - Korean Multitask Language Understanding (45 domains)",
    }
}


def get_metric_desc(lang: str = "ko") -> dict:
    """Get metric descriptions for language"""
    return METRIC_DESCRIPTIONS.get(lang, METRIC_DESCRIPTIONS["ko"])


def get_task_desc(task: str, lang: str = "ko") -> str:
    """Get task description for language"""
    return TASK_DESCRIPTIONS.get(lang, TASK_DESCRIPTIONS["ko"]).get(task, task)

# 스타일 - 메트릭 카드 텍스트 색상 수정
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    /* 메트릭 카드 스타일 수정 */
    [data-testid="stMetricValue"] {
        color: #1f2937 !important;
        font-size: 2rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: #4b5563 !important;
    }
    [data-testid="stMetricDelta"] {
        color: #059669 !important;
    }
    .info-box {
        background-color: #e8f4f8;
        border-left: 4px solid #1f77b4;
        padding: 10px 15px;
        margin: 10px 0;
        border-radius: 0 5px 5px 0;
    }
    /* 벤치마크 태그 스타일 */
    .benchmark-tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8em;
        font-weight: bold;
        color: white;
    }
    .benchmark-kobest { background-color: #3498db; }
    .benchmark-haerae { background-color: #27ae60; }
    .benchmark-kmmlu { background-color: #9b59b6; }
</style>
""", unsafe_allow_html=True)


def load_results(results_dir: str = "results") -> Dict[str, Any]:
    """결과 파일 로드 (항상 raw 파일에서 최신 결과 읽기)"""
    results_path = Path(results_dir)

    # 항상 개별 파일에서 로드 (최신 결과 보장)
    results = {"models": [], "benchmarks": [], "results": {}}
    raw_path = results_path / "raw"

    if raw_path.exists():
        for file in raw_path.glob("*.json"):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    model_name = data.get("model_name", "unknown")
                    task_name = data.get("task_name", "unknown")

                    if model_name not in results["models"]:
                        results["models"].append(model_name)
                    if task_name not in results["benchmarks"]:
                        results["benchmarks"].append(task_name)

                    if model_name not in results["results"]:
                        results["results"][model_name] = {}
                    results["results"][model_name][task_name] = data.get("metrics", {})
            except Exception as e:
                st.warning(f"파일 로드 실패: {file.name} - {e}")
                continue

    return results


def get_primary_metric(metrics: Dict[str, float]) -> tuple:
    """주요 메트릭과 값 반환"""
    priority = ["accuracy", "f1", "exact_match", "pearson", "micro_f1", "macro_f1"]
    for metric in priority:
        if metric in metrics and isinstance(metrics[metric], (int, float)):
            return metric, metrics[metric]
    # 첫 번째 숫자형 메트릭 반환
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            return k, v
    return None, None


def create_comparison_table(results: Dict[str, Any], lang: str = "ko") -> pd.DataFrame:
    """모델 간 비교 테이블 생성 (벤치마크 컬럼 포함)"""
    models = sorted(results.get("results", {}).keys())
    all_tasks = set()
    for tasks in results.get("results", {}).values():
        all_tasks.update(tasks.keys())

    # 벤치마크별로 정렬
    all_tasks = sorted(all_tasks, key=lambda x: (TASK_TO_BENCHMARK.get(x, "ZZZ"), x))

    # 데이터 구성
    rows = []
    for task in all_tasks:
        benchmark = TASK_TO_BENCHMARK.get(task, "Other" if lang == "en" else "기타")
        task_desc = get_task_desc(task, lang)
        desc = task_desc.split(" - ")[1] if " - " in task_desc else task_desc

        row = {
            "벤치마크": benchmark,
            "태스크": task,
            "설명": desc,
        }

        for model in models:
            metrics = results["results"].get(model, {}).get(task, {})
            metric_name, value = get_primary_metric(metrics)
            if value is not None:
                row[model] = value
            else:
                row[model] = None
        rows.append(row)

    return pd.DataFrame(rows)


def create_leaderboard(results: Dict[str, Any]) -> pd.DataFrame:
    """리더보드 데이터프레임 생성"""
    rows = []

    for model_name, tasks in results.get("results", {}).items():
        scores = []
        for task_name, metrics in tasks.items():
            metric_name, score = get_primary_metric(metrics)
            if score is not None:
                scores.append(score)

        if scores:
            avg_score = sum(scores) / len(scores)
            max_score = max(scores)
            min_score = min(scores)
            rows.append({
                "모델": model_name,
                "평균 점수": avg_score,
                "최고 점수": max_score,
                "최저 점수": min_score,
                "태스크 수": len(scores),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("평균 점수", ascending=False)
        df["순위"] = range(1, len(df) + 1)
        df = df[["순위", "모델", "평균 점수", "최고 점수", "최저 점수", "태스크 수"]]

    return df


def get_model_wins(results: Dict[str, Any]) -> Dict[str, int]:
    """각 모델이 1위를 차지한 태스크 수 계산"""
    models = list(results.get("results", {}).keys())
    wins = {m: 0 for m in models}

    all_tasks = set()
    for tasks in results.get("results", {}).values():
        all_tasks.update(tasks.keys())

    for task in all_tasks:
        best_score = -1
        best_model = None
        for model in models:
            metrics = results["results"].get(model, {}).get(task, {})
            _, score = get_primary_metric(metrics)
            if score is not None and score > best_score:
                best_score = score
                best_model = model
        if best_model:
            wins[best_model] += 1

    return wins


def create_radar_chart(results: Dict[str, Any], benchmarks: List[str], lang: str = "ko") -> go.Figure:
    """레이더 차트 생성"""
    fig = go.Figure()

    colors = px.colors.qualitative.Set2

    for idx, (model_name, tasks) in enumerate(results.get("results", {}).items()):
        scores = []
        labels = []

        for benchmark in benchmarks:
            if benchmark in tasks:
                metrics = tasks[benchmark]
                _, score = get_primary_metric(metrics)
                if score is not None:
                    scores.append(score)
                    labels.append(benchmark)

        if scores:
            # 닫힌 형태로 만들기
            scores_closed = scores + [scores[0]]
            labels_closed = labels + [labels[0]]

            fig.add_trace(go.Scatterpolar(
                r=scores_closed,
                theta=labels_closed,
                fill='toself',
                name=model_name,
                line=dict(color=colors[idx % len(colors)]),
            ))

    title = "벤치마크별 성능 비교 (레이더 차트)" if lang == "ko" else "Performance by Benchmark (Radar Chart)"
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
            )
        ),
        showlegend=True,
        title=title,
        height=500,
    )

    return fig


def create_heatmap(results: Dict[str, Any], lang: str = "ko") -> go.Figure:
    """히트맵 생성 (벤치마크별 정렬)"""
    models = sorted(results.get("results", {}).keys())
    tasks = set()
    for model_tasks in results.get("results", {}).values():
        tasks.update(model_tasks.keys())
    # 벤치마크별 정렬
    tasks = sorted(list(tasks), key=lambda x: (TASK_TO_BENCHMARK.get(x, "ZZZ"), x))

    # 데이터 매트릭스 생성
    z = []
    for model in models:
        row = []
        for task in tasks:
            metrics = results["results"].get(model, {}).get(task, {})
            _, score = get_primary_metric(metrics)
            row.append(score if score is not None else None)
        z.append(row)

    # 태스크 레이블에 벤치마크 표시
    task_labels = [f"{t}<br>({TASK_TO_BENCHMARK.get(t, '?')})" for t in tasks]

    colorbar_title = "점수" if lang == "ko" else "Score"
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=task_labels,
        y=models,
        colorscale="RdYlGn",
        zmin=0,
        zmax=1,
        text=[[f"{v:.2f}" if v is not None else "-" for v in row] for row in z],
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False,
        colorbar=dict(title=colorbar_title),
    ))

    title = "모델-태스크 성능 매트릭스 (히트맵)" if lang == "ko" else "Model-Task Performance Matrix (Heatmap)"
    xaxis_title = "태스크 (벤치마크)" if lang == "ko" else "Task (Benchmark)"
    yaxis_title = "모델" if lang == "ko" else "Model"

    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=400,
    )

    return fig


def create_grouped_bar_chart(results: Dict[str, Any], group_by_benchmark: bool = False, lang: str = "ko") -> go.Figure:
    """모델별 전체 태스크 비교 바 차트"""
    data = []

    # 컬럼명
    model_col = "모델" if lang == "ko" else "Model"
    task_col = "태스크" if lang == "ko" else "Task"
    benchmark_col = "벤치마크" if lang == "ko" else "Benchmark"
    score_col = "점수" if lang == "ko" else "Score"

    for model_name, tasks in results.get("results", {}).items():
        for task_name, metrics in tasks.items():
            _, score = get_primary_metric(metrics)
            if score is not None:
                benchmark = TASK_TO_BENCHMARK.get(task_name, "Other" if lang == "en" else "기타")
                data.append({
                    model_col: model_name,
                    task_col: task_name,
                    benchmark_col: benchmark,
                    score_col: score,
                })

    if not data:
        return go.Figure()

    df = pd.DataFrame(data)

    # 벤치마크별 정렬
    df["_sort"] = df[task_col].map(lambda x: (TASK_TO_BENCHMARK.get(x, "ZZZ"), x))
    df = df.sort_values("_sort")

    title = "태스크별 모델 성능 비교" if lang == "ko" else "Model Performance by Task"
    title_sorted = f"{title} (벤치마크별 정렬)" if lang == "ko" else f"{title} (Sorted by Benchmark)"

    if group_by_benchmark:
        fig = px.bar(
            df,
            x=task_col,
            y=score_col,
            color=model_col,
            barmode="group",
            title=title,
            height=500,
            facet_col=benchmark_col,
            facet_col_wrap=2,
        )
    else:
        fig = px.bar(
            df,
            x=task_col,
            y=score_col,
            color=model_col,
            barmode="group",
            title=title_sorted,
            height=500,
        )

    fig.update_layout(xaxis_tickangle=-45)

    return fig


def create_bar_chart(results: Dict[str, Any], selected_benchmark: str, lang: str = "ko") -> go.Figure:
    """특정 벤치마크의 바 차트 생성"""
    data = []

    # 컬럼명
    model_col = "모델" if lang == "ko" else "Model"
    metric_col = "메트릭" if lang == "ko" else "Metric"
    score_col = "점수" if lang == "ko" else "Score"

    for model_name, tasks in results.get("results", {}).items():
        if selected_benchmark in tasks:
            metrics = tasks[selected_benchmark]
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    data.append({
                        model_col: model_name,
                        metric_col: metric_name,
                        score_col: value,
                    })

    if not data:
        return go.Figure()

    df = pd.DataFrame(data)
    title = f"{selected_benchmark} 벤치마크 결과" if lang == "ko" else f"{selected_benchmark} Benchmark Results"
    fig = px.bar(
        df,
        x=model_col,
        y=score_col,
        color=metric_col,
        barmode="group",
        title=title,
    )

    return fig


def main():
    # 언어 선택 (사이드바 최상단)
    lang = st.sidebar.selectbox(
        "🌐 Language / 언어",
        options=["ko", "en"],
        format_func=lambda x: "한국어" if x == "ko" else "English",
        index=0,
    )

    T = lambda key: get_text(key, lang)

    st.title(T("title"))

    # 결과 로드
    results = load_results()

    if not results.get("results"):
        st.warning(T("no_results"))
        st.code("python main.py --models all --benchmarks all --limit 100")
        return

    # 사이드바
    st.sidebar.markdown("---")
    st.sidebar.header(T("filter"))
    models = sorted(results.get("results", {}).keys())
    selected_models = st.sidebar.multiselect(
        T("select_model"),
        options=models,
        default=models,
    )

    all_benchmarks = set()
    for tasks in results.get("results", {}).values():
        all_benchmarks.update(tasks.keys())
    benchmarks = sorted(list(all_benchmarks))

    # 벤치마크 카테고리 필터
    st.sidebar.markdown("---")
    benchmark_categories = ["KoBEST", "HAE-RAE", "KMMLU"]
    selected_categories = st.sidebar.multiselect(
        T("benchmark_category"),
        options=benchmark_categories,
        default=benchmark_categories,
    )

    # 사이드바에 메트릭 설명 추가
    st.sidebar.markdown("---")
    with st.sidebar.expander(T("metric_desc")):
        metric_descs = get_metric_desc(lang)
        for metric, desc in metric_descs.items():
            st.markdown(f"**{metric}**: {desc.split(' - ')[1]}")

    # 탭 구성
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        T("tab_comparison"), T("tab_leaderboard"), T("tab_visualization"),
        T("tab_details"), T("tab_info")
    ])

    # 필터링된 결과
    filtered_results = {"results": {}}
    for m in selected_models:
        if m in results.get("results", {}):
            filtered_results["results"][m] = {}
            for task, metrics in results["results"][m].items():
                task_category = TASK_TO_BENCHMARK.get(task, "기타")
                if task_category in selected_categories:
                    filtered_results["results"][m][task] = metrics

    with tab1:
        st.header(T("comparison_header"))

        st.markdown(f"""
        > {T("comparison_desc")}
        """)

        # 색상 범례
        st.markdown(f"""
        <div style="display: flex; gap: 20px; margin-bottom: 15px; flex-wrap: wrap;">
            <span style="background-color: #2ecc71; color: white; padding: 4px 12px; border-radius: 4px; font-weight: bold;">
                {T("best_score")}
            </span>
            <span style="background-color: #e74c3c22; padding: 4px 12px; border-radius: 4px;">
                {T("worst_score")}
            </span>
            <span style="padding: 4px 12px; border-radius: 4px; border: 1px solid #ddd;">
                {T("score_range")}
            </span>
        </div>
        """, unsafe_allow_html=True)

        # 범례 표시
        legend_html = " ".join([
            f'<span class="benchmark-tag benchmark-{cat.lower().replace("-", "")}">{cat}</span>'
            for cat in benchmark_categories
        ])
        st.markdown(f"**{T('benchmark_legend')}** {legend_html}", unsafe_allow_html=True)

        # 비교 테이블
        comparison_df = create_comparison_table(filtered_results, lang)
        if not comparison_df.empty:
            # 컬럼 번역
            display_df = translate_columns(comparison_df, lang)
            benchmark_col = COLUMN_TRANSLATIONS[lang]["벤치마크"]
            task_col = COLUMN_TRANSLATIONS[lang]["태스크"]
            desc_col = COLUMN_TRANSLATIONS[lang]["설명"]

            # 모델 컬럼만 포맷팅
            model_cols = [col for col in display_df.columns if col not in [benchmark_col, task_col, desc_col]]

            # 행별 최고점 강조 함수 (각 태스크별 1등 모델 강조)
            def highlight_row_best(row):
                """각 행(태스크)에서 최고 점수를 가진 모델 강조"""
                styles = [''] * len(row)
                # 숫자 값만 필터링
                numeric_vals = []
                numeric_indices = []
                for i, (col, val) in enumerate(row.items()):
                    if col in model_cols and pd.notna(val) and isinstance(val, (int, float)):
                        numeric_vals.append(val)
                        numeric_indices.append(i)

                if numeric_vals:
                    max_val = max(numeric_vals)
                    min_val = min(numeric_vals)
                    for i, val in zip(numeric_indices, numeric_vals):
                        if val == max_val:
                            styles[i] = 'background-color: #2ecc71; color: white; font-weight: bold'
                        elif val == min_val and max_val != min_val:
                            styles[i] = 'background-color: #e74c3c22'
                return styles

            def color_benchmark(val):
                colors = {
                    "KoBEST": "background-color: #3498db33; font-weight: bold",
                    "HAE-RAE": "background-color: #27ae6033; font-weight: bold",
                    "KMMLU": "background-color: #9b59b633; font-weight: bold",
                }
                return colors.get(val, "")

            styled_df = display_df.style.apply(
                highlight_row_best, axis=1
            ).format(
                {col: "{:.4f}" for col in model_cols}, na_rep="-"
            ).map(
                color_benchmark, subset=[benchmark_col]
            )

            st.dataframe(styled_df, height=600)

            # CSV 다운로드
            csv = comparison_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                T("download_csv"),
                csv,
                "benchmark_comparison.csv",
                "text/csv",
            )
        else:
            st.info(T("no_data"))

        # 전체 바 차트
        st.subheader(T("task_comparison"))
        bar_fig = create_grouped_bar_chart(filtered_results, lang=lang)
        st.plotly_chart(bar_fig, key="grouped_bar")

    with tab2:
        st.header(T("leaderboard_header"))

        # 메트릭 카드 - 컬럼 배경색 추가
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center;">
                <p style="color: #6c757d; margin: 0; font-size: 0.9em;">{T("num_models")}</p>
                <p style="color: #212529; margin: 0; font-size: 2em; font-weight: bold;">{len(selected_models)}</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            filtered_tasks = set()
            for tasks in filtered_results.get("results", {}).values():
                filtered_tasks.update(tasks.keys())
            st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center;">
                <p style="color: #6c757d; margin: 0; font-size: 0.9em;">{T("num_tasks")}</p>
                <p style="color: #212529; margin: 0; font-size: 2em; font-weight: bold;">{len(filtered_tasks)}</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            total_evals = sum(
                len(tasks) for tasks in filtered_results.get("results", {}).values()
            )
            st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; text-align: center;">
                <p style="color: #6c757d; margin: 0; font-size: 0.9em;">{T("total_evals")}</p>
                <p style="color: #212529; margin: 0; font-size: 2em; font-weight: bold;">{total_evals}</p>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            wins = get_model_wins(filtered_results)
            best_model = max(wins, key=wins.get) if wins else "-"
            st.markdown(f"""
            <div style="background-color: #d4edda; padding: 20px; border-radius: 10px; text-align: center;">
                <p style="color: #155724; margin: 0; font-size: 0.9em;">{T("most_wins")}</p>
                <p style="color: #155724; margin: 0; font-size: 1.5em; font-weight: bold;">{best_model}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 모델별 1위 횟수 표시
        st.subheader(T("model_wins"))
        wins = get_model_wins(filtered_results)
        wins_df = pd.DataFrame([
            {T("col_model"): model, T("col_wins"): count, T("col_ratio"): f"{count/len(filtered_tasks)*100:.1f}%" if filtered_tasks else "0%"}
            for model, count in sorted(wins.items(), key=lambda x: -x[1])
        ])
        if not wins_df.empty:
            st.dataframe(wins_df, height=150)

        # 리더보드
        st.subheader(T("overall_ranking"))
        st.markdown(f"> {T('ranking_desc')}")

        leaderboard = create_leaderboard(filtered_results)
        if not leaderboard.empty:
            # 컬럼 번역
            leaderboard_display = translate_columns(leaderboard, lang)
            rank_col = COLUMN_TRANSLATIONS[lang]["순위"]
            avg_col = COLUMN_TRANSLATIONS[lang]["평균 점수"]
            max_col = COLUMN_TRANSLATIONS[lang]["최고 점수"]
            min_col = COLUMN_TRANSLATIONS[lang]["최저 점수"]

            def highlight_first_place(s):
                if s.name == rank_col:
                    return ['background-color: #f1c40f; color: black; font-weight: bold' if v == 1 else '' for v in s]
                return ['' for _ in s]

            st.dataframe(
                leaderboard_display.style.format({
                    avg_col: "{:.4f}",
                    max_col: "{:.4f}",
                    min_col: "{:.4f}",
                }).apply(highlight_first_place).background_gradient(
                    subset=[avg_col], cmap="RdYlGn", vmin=0, vmax=1
                ),
                height=200,
            )
        else:
            st.info(T("no_data"))

        # 카테고리별 성능
        st.subheader(T("category_performance"))

        benchmark_task_map = {
            "KoBEST": ["boolq", "copa", "wic", "hellaswag", "sentineg"],
            "HAE-RAE": ["general_knowledge", "history", "reading_comprehension",
                       "standard_nomenclature", "loan_words", "rare_words"],
            "KMMLU": ["kmmlu"],
        }

        category_scores = []
        for model_name, tasks in filtered_results.get("results", {}).items():
            for category, category_tasks in benchmark_task_map.items():
                if category not in selected_categories:
                    continue
                scores = []
                for task in category_tasks:
                    if task in tasks:
                        _, score = get_primary_metric(tasks[task])
                        if score is not None:
                            scores.append(score)
                if scores:
                    category_scores.append({
                        "모델": model_name,
                        "카테고리": category,
                        "평균 점수": sum(scores) / len(scores),
                        "태스크 수": len(scores),
                    })

        if category_scores:
            cat_df = pd.DataFrame(category_scores)
            cat_display_df = translate_columns(cat_df, lang)
            model_col = COLUMN_TRANSLATIONS[lang]["모델"]
            cat_col = COLUMN_TRANSLATIONS[lang]["카테고리"]
            avg_score_col = COLUMN_TRANSLATIONS[lang]["평균 점수"]

            # 피벗 테이블로 변환
            pivot_df = cat_display_df.pivot(index=cat_col, columns=model_col, values=avg_score_col)

            # 각 행에서 최대값 강조
            def highlight_row_max(row):
                is_max = row == row.max()
                return [
                    'background-color: #2ecc71; color: white; font-weight: bold' if v else ''
                    for v in is_max
                ]

            st.dataframe(
                pivot_df.style.format("{:.4f}").apply(highlight_row_max, axis=1)
            )

            chart_title = "벤치마크 카테고리별 평균 점수" if lang == "ko" else "Average Score by Benchmark Category"
            cat_fig = px.bar(
                cat_display_df,
                x=cat_col,
                y=avg_score_col,
                color=model_col,
                barmode="group",
                title=chart_title,
                text=cat_display_df[avg_score_col].apply(lambda x: f"{x:.2f}"),
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            cat_fig.update_traces(textposition='outside')
            cat_fig.update_layout(yaxis_range=[0, 1.1])
            st.plotly_chart(cat_fig, key="category_bar")

    with tab3:
        st.header(T("tab_visualization"))

        # 레이더 차트
        st.subheader(T("radar_chart"))
        st.markdown(f"> {T('radar_desc')}")

        filtered_benchmarks = [b for b in benchmarks if TASK_TO_BENCHMARK.get(b, "기타") in selected_categories]

        if len(selected_models) > 0 and len(filtered_benchmarks) > 0:
            radar_fig = create_radar_chart(filtered_results, filtered_benchmarks[:12], lang)
            st.plotly_chart(radar_fig, key="radar")
        else:
            st.info(T("no_data"))

        # 히트맵
        st.subheader(T("heatmap"))
        st.markdown(f"> {T('heatmap_desc')}")

        heatmap_fig = create_heatmap(filtered_results, lang)
        st.plotly_chart(heatmap_fig, key="heatmap")

        # 벤치마크별 상세
        st.subheader(T("benchmark_detail"))

        def format_benchmark_option(x):
            desc = get_task_desc(x, lang)
            short_desc = desc.split(' - ')[0] if ' - ' in desc else desc
            return f"[{TASK_TO_BENCHMARK.get(x, '?')}] {x} - {short_desc}"

        selected_benchmark = st.selectbox(
            T("select_benchmark"),
            options=filtered_benchmarks,
            format_func=format_benchmark_option
        )
        if selected_benchmark:
            # 선택된 벤치마크 설명
            task_full_desc = get_task_desc(selected_benchmark, lang)
            st.info(f"**{TASK_TO_BENCHMARK.get(selected_benchmark, '?')}** > {task_full_desc}")

            bar_fig = create_bar_chart(filtered_results, selected_benchmark, lang)
            st.plotly_chart(bar_fig, key="benchmark_bar")

    with tab4:
        st.header(T("details_header"))

        for model_name in selected_models:
            if model_name not in filtered_results.get("results", {}):
                continue

            with st.expander(f"📦 {model_name}", expanded=True):
                tasks = filtered_results["results"][model_name]

                # 컬럼명 매핑
                benchmark_col = COLUMN_TRANSLATIONS[lang]["벤치마크"]
                task_col = COLUMN_TRANSLATIONS[lang]["태스크"]
                desc_col = COLUMN_TRANSLATIONS[lang]["설명"]

                # 테이블 형식
                rows = []
                for task_name, metrics in sorted(tasks.items(), key=lambda x: (TASK_TO_BENCHMARK.get(x[0], "ZZZ"), x[0])):
                    benchmark = TASK_TO_BENCHMARK.get(task_name, "Other" if lang == "en" else "기타")
                    task_full_desc = get_task_desc(task_name, lang)
                    task_short_desc = task_full_desc.split(" - ")[1] if " - " in task_full_desc else ""
                    row = {
                        benchmark_col: benchmark,
                        task_col: task_name,
                        desc_col: task_short_desc,
                    }
                    for k, v in metrics.items():
                        if isinstance(v, (int, float)):
                            row[k] = v
                    rows.append(row)

                if rows:
                    df = pd.DataFrame(rows)
                    # 숫자 컬럼 포맷팅
                    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

                    # 각 숫자 컬럼에서 최대값 강조
                    def highlight_col_max(s):
                        if s.dtype == 'float64' or s.dtype == 'int64':
                            is_max = s == s.max()
                            is_min = s == s.min()
                            styles = []
                            for i, (mx, mn) in enumerate(zip(is_max, is_min)):
                                if mx and not mn:  # 최대값 (최소값과 같지 않을 때만)
                                    styles.append('background-color: #2ecc71; color: white; font-weight: bold')
                                elif mn and len(s) > 1 and s.max() != s.min():  # 최소값
                                    styles.append('background-color: #e74c3c22')
                                else:
                                    styles.append('')
                            return styles
                        return ['' for _ in s]

                    st.dataframe(
                        df.style.format({col: "{:.4f}" for col in numeric_cols}).apply(
                            highlight_col_max, subset=list(numeric_cols)
                        ),
                        height=400,
                    )

    with tab5:
        st.header(T("info_header"))

        st.markdown(f"""
        ### {T("benchmark_categories")}

        {T("benchmark_intro")}
        """)

        col1, col2 = st.columns(2)

        # 언어별 벤치마크 설명
        if lang == "ko":
            with col1:
                st.markdown("""
                #### 🔵 KoBEST
                한국어 자연어 이해 벤치마크

                | 태스크 | 설명 | 메트릭 |
                |--------|------|--------|
                | BoolQ | 예/아니오 질의응답 | Accuracy |
                | COPA | 인과 추론 | Accuracy |
                | WiC | 단어 의미 구별 | Accuracy |
                | HellaSwag | 상식 기반 문장 완성 | Accuracy |
                | SentiNeg | 부정 표현 감정 분석 | Accuracy |
                """)

                st.markdown("""
                #### 🟢 HAE-RAE
                한국어/문화 특화 평가

                | 태스크 | 설명 | 메트릭 |
                |--------|------|--------|
                | General Knowledge | 일반 상식 | Accuracy |
                | History | 한국사 | Accuracy |
                | Reading Comprehension | 독해 | Accuracy |
                | Standard Nomenclature | 맞춤법 | Accuracy |
                | Loan Words | 외래어 표기 | Accuracy |
                | Rare Words | 희귀 단어 | Accuracy |
                """)

            with col2:
                st.markdown("""
                #### 🟣 KMMLU
                Korean Massive Multitask Language Understanding

                | 태스크 | 설명 | 메트릭 |
                |--------|------|--------|
                | KMMLU | 45개 분야 지식 평가 | Accuracy |

                *회계, 법학, 의학, 공학, 역사, 수학 등 45개 전문 분야*
                """)
        else:  # English
            with col1:
                st.markdown("""
                #### 🔵 KoBEST
                Korean NLU Benchmark

                | Task | Description | Metric |
                |------|-------------|--------|
                | BoolQ | Yes/No Question Answering | Accuracy |
                | COPA | Causal Reasoning | Accuracy |
                | WiC | Word Sense Disambiguation | Accuracy |
                | HellaSwag | Commonsense Completion | Accuracy |
                | SentiNeg | Negation Sentiment | Accuracy |
                """)

                st.markdown("""
                #### 🟢 HAE-RAE
                Korean Culture-Specific Evaluation

                | Task | Description | Metric |
                |------|-------------|--------|
                | General Knowledge | Korean Culture & Common Sense | Accuracy |
                | History | Korean Historical Facts | Accuracy |
                | Reading Comprehension | Text Understanding | Accuracy |
                | Standard Nomenclature | Spelling/Grammar | Accuracy |
                | Loan Words | Foreign Word Notation | Accuracy |
                | Rare Words | Vocabulary Assessment | Accuracy |
                """)

            with col2:
                st.markdown("""
                #### 🟣 KMMLU
                Korean Massive Multitask Language Understanding

                | Task | Description | Metric |
                |------|-------------|--------|
                | KMMLU | 45-Domain Knowledge Evaluation | Accuracy |

                *Accounting, Law, Medicine, Engineering, History, Math, etc. (45 domains)*
                """)

        st.markdown("---")
        st.markdown(f"""
        ### {T("metric_explanation")}

        | {T("col_metric")} | {T("col_desc")} | {"범위" if lang == "ko" else "Range"} |
        |--------|------|------|
        | **Accuracy** | {"전체 예측 중 정답 비율" if lang == "ko" else "Ratio of correct predictions"} | 0~1 |
        | **F1 Score** | {"정밀도와 재현율의 조화평균" if lang == "ko" else "Harmonic mean of precision and recall"} | 0~1 |
        | **Pearson** | {"예측값과 실제값의 상관계수" if lang == "ko" else "Correlation between prediction and actual"} | -1~1 |
        | **Exact Match (EM)** | {"완전히 일치하는 비율" if lang == "ko" else "Ratio of exact matches"} | 0~1 |
        | **Joint Goal Accuracy (JGA)** | {"대화 상태 완전 일치 비율" if lang == "ko" else "Dialog state exact match ratio"} | 0~1 |
        """)

    # 푸터
    st.markdown("---")
    footer_text = (
        "Built with Streamlit | "
        f"{'AWS Bedrock 한국어 모델 벤치마크' if lang == 'ko' else 'AWS Bedrock Korean Model Benchmark'} | "
        f"{'모델' if lang == 'ko' else 'Models'}: {', '.join(models)}"
    )
    st.markdown(footer_text)


if __name__ == "__main__":
    main()
