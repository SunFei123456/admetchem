from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class UserBase(BaseModel):
    """用户基础模式"""

    # 用户邮箱
    email: EmailStr

    # 用户名
    username: str

    # 用户头像URL
    avatar: Optional[str] = None

    # 是否激活，默认为True
    is_active: bool = True


class UserCreate(UserBase):
    """创建用户模式"""

    # 密码
    password: str = Field(..., min_length=8, max_length=20, description="密码长度8-20位")


class UserLogin(BaseModel):
    """用户登录模式"""
    
    email: EmailStr
    password: str = Field(..., max_length=20, description="密码")


class UserUpdate(BaseModel):
    """更新用户模式"""

    # 用户名（可选）
    username: Optional[str] = Field(None, min_length=2, max_length=50)
    
    # 用户头像URL（可选）
    avatar: Optional[str] = None
    
    # 旧密码（修改密码时必填）
    old_password: Optional[str] = Field(None, max_length=20)
    
    # 新密码（可选）
    new_password: Optional[str] = Field(None, min_length=8, max_length=20)


class User(UserBase):
    """用户响应模式"""

    # 用户ID
    id: int

    # 创建时间
    created_at: datetime

    # 更新时间（可选）
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class Token(BaseModel):
    """Token响应模式"""
    
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Token数据模式"""
    
    user_id: Optional[int] = None
    email: Optional[str] = None


class DruglikenessRequest(BaseModel):
    """类药性评估请求模型"""

    smiles: str
    rules: Optional[List[str]] = None  # 已废弃，保留向后兼容
    selected_items: Optional[List[str]] = None  # 新版：用户选择的项目列表（规则+性质）
    
    class Config:
        json_schema_extra = {
            "example": {
                "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
                "selected_items": ["Lipinski", "QED", "SAscore"]
            }
        }


class DruglikenessResponse(BaseModel):
    """类药性评估响应模型（向后兼容旧版API）"""

    metrics: Dict[str, Dict[str, float]]
    matches: Dict[str, float]
    # total_score: float  # 暂时注释掉
    smiles: str


class ComprehensiveEvaluationResponse(BaseModel):
    """综合评估响应模型（新版API）"""
    
    selected_items: List[str]  # 用户选择的项目
    druglikeness_rules: Optional[Dict[str, Any]] = None  # 类药性规则结果
    molecular_properties: Optional[Dict[str, Any]] = None  # 分子性质结果
    smiles: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "selected_items": ["Lipinski", "QED", "SAscore"],
                "druglikeness_rules": {
                    "metrics": {"Lipinski": {"mw": 206.28, "logp": 3.5}},
                    "matches": {"Lipinski": 1.0},
                    "total_score": 1.0
                },
                "molecular_properties": {
                    "QED": 0.82,
                    "QED_status": "excellent"
                },
                "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
            }
        }


class SdfToSmilesResponse(BaseModel):
    """SDF转SMILES响应模型"""
    
    results: List[Dict[str, str]]  # 包含分子名称和SMILES的列表
    total_molecules: int  # 总分子数
    successful_conversions: int  # 成功转换的分子数
    failed_conversions: int  # 失败的分子数
