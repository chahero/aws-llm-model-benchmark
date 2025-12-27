"""
평가 메트릭 모듈
"""
from typing import List, Dict, Any, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from scipy.stats import pearsonr, spearmanr


def calculate_accuracy(
    predictions: List[Any],
    references: List[Any],
) -> float:
    """정확도 계산"""
    return accuracy_score(references, predictions)


def calculate_f1(
    predictions: List[Any],
    references: List[Any],
    average: str = "macro",
) -> float:
    """
    F1 점수 계산

    Args:
        predictions: 예측값 리스트
        references: 정답 리스트
        average: 평균 방식 (macro, micro, weighted, binary)

    Returns:
        float: F1 점수
    """
    return f1_score(references, predictions, average=average, zero_division=0)


def calculate_precision(
    predictions: List[Any],
    references: List[Any],
    average: str = "macro",
) -> float:
    """정밀도 계산"""
    return precision_score(references, predictions, average=average, zero_division=0)


def calculate_recall(
    predictions: List[Any],
    references: List[Any],
    average: str = "macro",
) -> float:
    """재현율 계산"""
    return recall_score(references, predictions, average=average, zero_division=0)


def calculate_exact_match(
    predictions: List[str],
    references: List[str],
    normalize: bool = True,
) -> float:
    """
    Exact Match 계산 (MRC용)

    Args:
        predictions: 예측 텍스트 리스트
        references: 정답 텍스트 리스트
        normalize: 정규화 여부

    Returns:
        float: EM 점수
    """
    if not predictions or not references:
        return 0.0

    def normalize_text(text: str) -> str:
        """텍스트 정규화"""
        return text.strip().lower()

    matches = 0
    for pred, ref in zip(predictions, references):
        if normalize:
            pred = normalize_text(pred)
            ref = normalize_text(ref)
        if pred == ref:
            matches += 1

    return matches / len(predictions)


def calculate_pearson_correlation(
    predictions: List[float],
    references: List[float],
) -> float:
    """
    피어슨 상관계수 계산 (STS용)

    Args:
        predictions: 예측 점수 리스트
        references: 정답 점수 리스트

    Returns:
        float: 피어슨 상관계수
    """
    if len(predictions) < 2:
        return 0.0
    correlation, _ = pearsonr(predictions, references)
    return correlation if not np.isnan(correlation) else 0.0


def calculate_spearman_correlation(
    predictions: List[float],
    references: List[float],
) -> float:
    """
    스피어만 상관계수 계산

    Args:
        predictions: 예측 점수 리스트
        references: 정답 점수 리스트

    Returns:
        float: 스피어만 상관계수
    """
    if len(predictions) < 2:
        return 0.0
    correlation, _ = spearmanr(predictions, references)
    return correlation if not np.isnan(correlation) else 0.0


def calculate_classification_metrics(
    predictions: List[Any],
    references: List[Any],
) -> Dict[str, float]:
    """
    분류 메트릭 종합 계산

    Returns:
        Dict: accuracy, precision, recall, f1 포함
    """
    return {
        "accuracy": calculate_accuracy(predictions, references),
        "precision": calculate_precision(predictions, references),
        "recall": calculate_recall(predictions, references),
        "f1": calculate_f1(predictions, references),
    }


def calculate_token_f1(
    prediction: str,
    reference: str,
) -> float:
    """
    토큰 단위 F1 계산 (MRC용)

    Args:
        prediction: 예측 텍스트
        reference: 정답 텍스트

    Returns:
        float: 토큰 F1 점수
    """
    pred_tokens = set(prediction.split())
    ref_tokens = set(reference.split())

    if not pred_tokens or not ref_tokens:
        return 0.0

    common = pred_tokens & ref_tokens

    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)


def calculate_mrc_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """
    MRC 메트릭 계산 (EM + F1)

    Returns:
        Dict: exact_match, f1 포함
    """
    em = calculate_exact_match(predictions, references)

    f1_scores = [
        calculate_token_f1(pred, ref)
        for pred, ref in zip(predictions, references)
    ]
    avg_f1 = np.mean(f1_scores) if f1_scores else 0.0

    return {
        "exact_match": em,
        "f1": avg_f1,
    }
