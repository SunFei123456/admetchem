from pydantic import BaseModel, EmailStr
from typing import Optional, Dict, Any, List
from datetime import datetime


class UserBase(BaseModel):
    """用户基础模式"""

    # 用户邮箱
    email: EmailStr

    # 用户名
    username: str

    # 是否激活，默认为True
    is_active: bool = True


class UserCreate(UserBase):
    """创建用户模式"""

    # 密码
    password: str


class UserUpdate(UserBase):
    """更新用户模式"""

    # 密码（可选）
    password: Optional[str] = None


class User(UserBase):
    """用户响应模式"""

    # 用户ID
    id: int

    # 创建时间
    created_at: datetime

    # 更新时间（可选）
    updated_at: Optional[datetime] = None

    class Config:
        orm_mode = True


class DruglikenessRequest(BaseModel):
    """类药性评估请求模型"""

    smiles: str
    rules: Optional[List[str]] = None  # 可选的规则列表，如果为空则返回所有规则


class DruglikenessResponse(BaseModel):
    """类药性评估响应模型"""

    metrics: Dict[str, Dict[str, float]]
    matches: Dict[str, float]
    # total_score: float  # 暂时注释掉
    smiles: str


class SdfToSmilesResponse(BaseModel):
    """SDF转SMILES响应模型"""
    
    results: List[Dict[str, str]]  # 包含分子名称和SMILES的列表
    total_molecules: int  # 总分子数
    successful_conversions: int  # 成功转换的分子数
    failed_conversions: int  # 失败的分子数
