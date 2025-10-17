from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.api.routes import router
from app.core.config import settings
from app.core.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时执行
    print("🚀 正在启动应用...")
    print("📊 正在同步数据库模型...")
    init_db()  # 自动创建/更新表结构
    print("✅ 数据库同步完成！")
    yield
    # 关闭时执行
    print("👋 应用正在关闭...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="ADMET Chemistry Server API",
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix=settings.API_V1_STR)


@app.get("/")
async def root():
    return {"message": "ADMET Chemistry Server API"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
