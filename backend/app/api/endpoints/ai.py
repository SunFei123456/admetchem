"""
AI 聊天 API 端点
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from typing import List, AsyncIterator
import json

from app.core.database import get_db
from app.core.auth import get_current_user
from app.models.models import User
from app.services.deepseek_service import DeepSeekService
from app.utils.response import success, fail
from pydantic import BaseModel


router = APIRouter(tags=["AI Chat"])


class ChatRequestAPI(BaseModel):
    """API 聊天请求模型"""
    model: str = None  # 可选，默认使用配置中的模型
    messages: List[dict]  # [{"role": "user", "content": "..."}]
    temperature: float = None  # 可选，默认使用配置中的温度


class AIProviderInfo(BaseModel):
    """AI 服务提供商信息"""
    name: str
    models: List[str]
    description: str


@router.post("/chat", summary="发送聊天消息")
async def chat(
    request: ChatRequestAPI,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    发送聊天消息到 DeepSeek AI（从服务器配置读取 API Key）
    
    - **model**: 模型名称（可选，默认 deepseek-chat）
    - **messages**: 聊天消息列表
    - **temperature**: 温度参数（可选，默认 0.7）
    """
    try:
        # 直接调用 DeepSeek 服务
        response = await DeepSeekService.chat(
            messages=request.messages,
            model=request.model,
            temperature=request.temperature
        )

        # TODO: 保存对话历史到数据库

        return success(data=response)

    except ValueError as e:
        # API Key 未配置
        return fail(
            message=str(e),
            code=40002,
            status_code=400
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [AI] Chat error: {str(e)}")
        return fail(
            message=f"AI 服务调用失败: {str(e)}",
            code=50001,
            status_code=500
        )


@router.get("/providers", summary="获取 AI 服务配置信息")
async def get_providers():
    """获取 DeepSeek 服务配置信息"""
    from app.core.config import settings
    
    providers_info = {
        "deepseek": {
            "name": "DeepSeek",
            "description": "性价比最高，适合大量使用，新用户送 500 万 tokens",
            "models": DeepSeekService.get_available_models(),
            "register_url": "https://platform.deepseek.com/",
            "configured": DeepSeekService.is_configured(),
            "current_model": settings.DEEPSEEK_MODEL,
            "current_temperature": settings.DEEPSEEK_TEMPERATURE
        }
    }

    return success(data=providers_info)


@router.get("/models", summary="获取可用模型列表")
async def get_models():
    """获取 DeepSeek 可用模型列表"""
    
    models = DeepSeekService.get_available_models()
    return success(data=models)


@router.post("/chat/stream", summary="发送聊天消息（流式响应）")
async def chat_stream(
    request: ChatRequestAPI,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    发送聊天消息到 DeepSeek AI（流式响应，从服务器配置读取 API Key）
    
    - **model**: 模型名称（可选，默认 deepseek-chat）
    - **messages**: 聊天消息列表
    - **temperature**: 温度参数（可选，默认 0.7）
    """
    try:
        # 创建流式生成器
        async def event_generator() -> AsyncIterator[str]:
            try:
                # 直接调用 DeepSeek 流式服务
                async for content_chunk in DeepSeekService.chat_stream(
                    messages=request.messages,
                    model=request.model,
                    temperature=request.temperature
                ):
                    yield f"data: {json.dumps({'content': content_chunk})}\n\n"
                
                yield f"data: {json.dumps({'done': True})}\n\n"
                    
            except Exception as e:
                print(f"❌ [AI Stream] Error: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ [AI Stream] Setup error: {str(e)}")
        
        async def error_generator():
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return StreamingResponse(
            error_generator(),
            media_type="text/event-stream"
        )

