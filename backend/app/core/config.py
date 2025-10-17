from typing import Any, Dict, List, Optional, Union
from pydantic_settings import BaseSettings
import json


class Settings(BaseSettings):
    PROJECT_NAME: str = "ADMETchem_server"
    API_V1_STR: str = "/api/v1"

    # CORS（允许在 .env 中使用逗号分隔字符串或 JSON 数组）
    # 这里将类型定义为 str，避免 pydantic-settings 对复杂类型进行 JSON 解码导致报错
    BACKEND_CORS_ORIGINS: str = (
        "http://localhost:3000,http://localhost:3001,http://localhost:8080"
    )

    def cors_origins_list(self) -> List[str]:
        v = self.BACKEND_CORS_ORIGINS or ""
        v = v.strip()
        if not v:
            return []
        # 若为 JSON 数组则优先按 JSON 解析
        if v.startswith("["):
            try:
                data = json.loads(v)
                # 仅保留字符串项
                return [str(i).strip() for i in data if str(i).strip()]
            except Exception:
                # 回退到逗号分隔解析
                pass
        # 逗号分隔解析
        return [i.strip() for i in v.split(",") if i.strip()]

    # Database
    DATABASE_URL: str = "mysql+pymysql://root:123456@localhost/admetchem_db"

    # JWT
    SECRET_KEY: str = "your-secret-key-here"
    # 使用HS256算法
    ALGORITHM: str = "HS256"
    # token过期失常30天
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 30

    # DeepSeek AI 配置
    DEEPSEEK_API_KEY: str = ""  # 从环境变量或 .env 文件读取
    DEEPSEEK_MODEL: str = "deepseek-chat"  # 默认模型: deepseek-chat
    DEEPSEEK_TEMPERATURE: float = 0.7  # 默认温度 (0-1)

    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()
