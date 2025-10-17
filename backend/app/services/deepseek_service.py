"""
DeepSeek AI 服务（简化版）
官方文档: https://platform.deepseek.com/api-docs/
"""
import httpx
import json
from typing import List, Dict, AsyncIterator
from app.core.config import settings


class DeepSeekService:
    """DeepSeek AI 服务（直接使用配置）"""

    BASE_URL = "https://api.deepseek.com/v1/chat/completions"
    
    AVAILABLE_MODELS = [
        "deepseek-chat",
        "deepseek-coder"
    ]

    @classmethod
    async def chat(
        cls,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = None,
        max_tokens: int = None
    ) -> Dict:
        """
        发送聊天请求（非流式）
        
        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            model: 模型名称（默认使用配置）
            temperature: 温度参数（默认使用配置）
            max_tokens: 最大 token 数（可选）
            
        Returns:
            Dict: 响应数据
        """
        api_key = settings.DEEPSEEK_API_KEY
        if not api_key:
            raise ValueError("DeepSeek API Key 未配置，请在 .env 文件中设置 DEEPSEEK_API_KEY")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model or settings.DEEPSEEK_MODEL,
            "messages": messages,
            "temperature": temperature if temperature is not None else settings.DEEPSEEK_TEMPERATURE,
            "stream": False
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                cls.BASE_URL,
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            data = response.json()

        # 返回响应数据
        return {
            "content": data["choices"][0]["message"]["content"],
            "model": data["model"],
            "usage": data.get("usage"),
            "finish_reason": data["choices"][0].get("finish_reason")
        }

    @classmethod
    async def chat_stream(
        cls,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = None,
        max_tokens: int = None
    ) -> AsyncIterator[str]:
        """
        发送聊天请求（流式响应）
        
        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            model: 模型名称（默认使用配置）
            temperature: 温度参数（默认使用配置）
            max_tokens: 最大 token 数（可选）
            
        Yields:
            str: 流式返回的文本片段
        """
        api_key = settings.DEEPSEEK_API_KEY
        if not api_key:
            raise ValueError("DeepSeek API Key 未配置，请在 .env 文件中设置 DEEPSEEK_API_KEY")
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model or settings.DEEPSEEK_MODEL,
            "messages": messages,
            "temperature": temperature if temperature is not None else settings.DEEPSEEK_TEMPERATURE,
            "stream": True
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                cls.BASE_URL,
                json=payload,
                headers=headers
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # 移除 "data: " 前缀
                        
                        if data_str == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(data_str)
                            
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue

    @classmethod
    def get_available_models(cls) -> List[str]:
        """获取可用模型列表"""
        return cls.AVAILABLE_MODELS

    @classmethod
    def is_configured(cls) -> bool:
        """检查是否已配置 API Key"""
        return bool(settings.DEEPSEEK_API_KEY)

