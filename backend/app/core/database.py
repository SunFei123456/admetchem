from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

engine = create_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """创建所有表（如果不存在）"""
    # 确保在创建表之前已加载所有模型
    # 这样 Base.metadata 里会包含模型元数据
    import app.models.models  # noqa: F401

    Base.metadata.create_all(bind=engine)


def init_db():
    """初始化数据库 - 应用启动时调用"""
    # 导入所有模型，确保它们被注册到 Base.metadata
    import app.models.models  # noqa: F401
    
    # 自动创建所有表和字段
    Base.metadata.create_all(bind=engine)
