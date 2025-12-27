"""
Nova 모델 래퍼
"""
from typing import Dict, Any, List, Optional

from config.settings import MODELS, model_config
from src.models.bedrock_client import BedrockClient, ModelResponse


class NovaModel:
    """Nova 모델 래퍼 클래스"""

    def __init__(self, model_name: str, region: str = None):
        """
        Args:
            model_name: 모델 이름 (nova-micro, nova-lite, nova-pro)
            region: AWS 리전
        """
        if model_name not in MODELS:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available models: {list(MODELS.keys())}"
            )

        self.model_name = model_name
        self.model_id = MODELS[model_name]
        self.client = BedrockClient(region)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
    ) -> ModelResponse:
        """
        텍스트 생성

        Args:
            prompt: 사용자 프롬프트
            system_prompt: 시스템 프롬프트
            max_tokens: 최대 토큰 수
            temperature: 온도 파라미터
            top_p: Top-p 파라미터

        Returns:
            ModelResponse: 모델 응답
        """
        return self.client.invoke_with_text(
            model_id=self.model_id,
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def batch_generate(
        self,
        prompts: List[str],
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> List[ModelResponse]:
        """
        배치 텍스트 생성

        Args:
            prompts: 프롬프트 리스트
            system_prompt: 시스템 프롬프트
            **kwargs: 추가 파라미터

        Returns:
            List[ModelResponse]: 응답 리스트
        """
        results = []
        for prompt in prompts:
            result = self.generate(prompt, system_prompt, **kwargs)
            results.append(result)
        return results

    def __repr__(self) -> str:
        return f"NovaModel(name={self.model_name}, id={self.model_id})"


def get_all_models(region: str = None) -> Dict[str, NovaModel]:
    """
    모든 Nova 모델 인스턴스 반환

    Args:
        region: AWS 리전

    Returns:
        Dict[str, NovaModel]: 모델 이름 -> NovaModel 매핑
    """
    return {name: NovaModel(name, region) for name in MODELS.keys()}
