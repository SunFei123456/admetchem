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
        from_attributes = True


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
