from sqlalchemy import Column, Integer, String, DateTime, Boolean
from sqlalchemy.sql import func
from app.core.database import Base


class User(Base):
    """用户模型"""

    __tablename__ = "users"

    # 用户ID，主键，自增
    id = Column(Integer, primary_key=True, index=True)

    # 用户邮箱，唯一，索引
    email = Column(String(254), unique=True, index=True, nullable=False)

    # 用户名，唯一，索引
    username = Column(String(50), unique=True, index=True, nullable=False)

    # 哈希密码，不可为空
    hashed_password = Column(String(255), nullable=False)

    # 用户头像URL，默认头像
    avatar = Column(
        String(500), 
        nullable=True,
        default="https://res.cloudinary.com/dazdjqzwd/image/upload/v1760685557/68d9f143293f4f17bccdbc791dcf779d_f0rcro.png"
    )

    # 用户是否激活，默认为True
    is_active = Column(Boolean, default=True)

    # 创建时间，自动设置为当前时间
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # 更新时间，更新时自动设置为当前时间
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
