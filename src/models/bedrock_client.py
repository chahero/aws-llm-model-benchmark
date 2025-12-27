"""
AWS Bedrock Runtime 클라이언트
"""
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

import boto3
from botocore.config import Config

from config.settings import aws_config, model_config


@dataclass
class ModelResponse:
    """모델 응답 데이터 클래스"""
    output: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    stop_reason: str
    raw_response: Dict[str, Any]


class BedrockClient:
    """AWS Bedrock Runtime 클라이언트"""

    def __init__(self, region: str = None):
        """
        Args:
            region: AWS 리전. 기본값은 설정에서 가져옴
        """
        self.region = region or aws_config.region

        # boto3 클라이언트 설정
        config = Config(
            region_name=self.region,
            retries={"max_attempts": 3, "mode": "adaptive"},
        )

        self.client = boto3.client(
            service_name="bedrock-runtime",
            config=config,
        )

    def invoke(
        self,
        model_id: str,
        messages: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
    ) -> ModelResponse:
        """
        Bedrock Converse API를 사용하여 모델 호출

        Args:
            model_id: 모델 ID (예: amazon.nova-pro-v1:0)
            messages: 대화 메시지 리스트
            system_prompt: 시스템 프롬프트 (선택)
            max_tokens: 최대 토큰 수
            temperature: 온도 파라미터
            top_p: Top-p 파라미터

        Returns:
            ModelResponse: 모델 응답
        """
        # 기본값 설정
        max_tokens = max_tokens or model_config.DEFAULT_MAX_TOKENS
        temperature = temperature if temperature is not None else model_config.DEFAULT_TEMPERATURE

        # 추론 설정 구성
        inference_config = {
            "maxTokens": max_tokens,
            "temperature": temperature,
        }

        # Claude 모델은 temperature와 top_p를 동시에 사용할 수 없음
        is_claude = "anthropic" in model_id.lower() or "claude" in model_id.lower()
        if not is_claude and top_p is not None:
            inference_config["topP"] = top_p
        elif not is_claude:
            inference_config["topP"] = model_config.DEFAULT_TOP_P

        # 요청 구성
        request_params = {
            "modelId": model_id,
            "messages": messages,
            "inferenceConfig": inference_config,
        }

        # 시스템 프롬프트 추가
        if system_prompt:
            request_params["system"] = [{"text": system_prompt}]

        # 시간 측정 및 API 호출
        start_time = time.perf_counter()
        response = self.client.converse(**request_params)
        latency_ms = (time.perf_counter() - start_time) * 1000

        # 응답 파싱
        output_message = response.get("output", {}).get("message", {})
        content = output_message.get("content", [])
        output_text = content[0].get("text", "") if content else ""

        usage = response.get("usage", {})

        return ModelResponse(
            output=output_text,
            input_tokens=usage.get("inputTokens", 0),
            output_tokens=usage.get("outputTokens", 0),
            latency_ms=latency_ms,
            stop_reason=response.get("stopReason", ""),
            raw_response=response,
        )

    def invoke_with_text(
        self,
        model_id: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> ModelResponse:
        """
        텍스트 프롬프트로 모델 호출 (편의 메서드)

        Args:
            model_id: 모델 ID
            prompt: 사용자 프롬프트
            system_prompt: 시스템 프롬프트
            **kwargs: 추가 파라미터

        Returns:
            ModelResponse: 모델 응답
        """
        messages = [
            {
                "role": "user",
                "content": [{"text": prompt}],
            }
        ]
        return self.invoke(model_id, messages, system_prompt, **kwargs)

    def test_connection(self) -> bool:
        """
        연결 테스트

        Returns:
            bool: 연결 성공 여부
        """
        try:
            # 간단한 요청으로 연결 테스트
            self.invoke_with_text(
                model_id=model_config.NOVA_MICRO,
                prompt="Hi",
                max_tokens=5,
            )
            return True
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False
