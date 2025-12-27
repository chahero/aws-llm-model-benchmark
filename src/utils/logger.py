"""
로깅 유틸리티
"""
import sys
from loguru import logger

# 기본 로거 설정
logger.remove()  # 기본 핸들러 제거

# 콘솔 출력 설정
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)

# 파일 출력 설정
logger.add(
    "logs/benchmark_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    rotation="1 day",
    retention="7 days",
    encoding="utf-8",
)


def get_logger(name: str = None):
    """로거 인스턴스 반환"""
    if name:
        return logger.bind(name=name)
    return logger
