"""
ADMET预测相关的Pydantic模型
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class ADMETPredictionRequest(BaseModel):
    """ADMET预测请求模型"""
    
    smiles: str = Field(..., description="SMILES字符串")
    selected_properties: List[str] = Field(
        ..., 
        description="选择的ADMET属性ID列表",
        min_length=1
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "smiles": "CCO",
                "selected_properties": ["cyp1a2", "logS", "ames", "bbb"]
            }
        }


class ADMETBatchPredictionRequest(BaseModel):
    """ADMET批量预测请求模型"""
    
    smiles_list: List[str] = Field(
        ..., 
        description="SMILES字符串列表",
        min_length=1
    )
    selected_properties: List[str] = Field(
        ..., 
        description="选择的ADMET属性ID列表",
        min_length=1
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "smiles_list": ["CCO", "CC(=O)O", "c1ccccc1"],
                "selected_properties": ["cyp1a2", "logS", "ames", "bbb"]
            }
        }


class ClassificationPrediction(BaseModel):
    """分类预测结果模型"""
    
    name: str = Field(..., description="任务名称")
    type: str = Field(default="classification", description="任务类型")
    probability: float = Field(..., description="预测概率 (0-1)")
    symbol: str = Field(..., description="ADMET符号 (---, --, -, +, ++, +++)")
    description: str = Field(..., description="符号描述")


class RegressionPrediction(BaseModel):
    """回归预测结果模型"""
    
    name: str = Field(..., description="任务名称")
    type: str = Field(default="regression", description="任务类型")
    value: float = Field(..., description="预测值")


class ADMETPredictionResult(BaseModel):
    """单个分子的ADMET预测结果"""
    
    smiles: str = Field(..., description="SMILES字符串")
    selected_properties: List[str] = Field(..., description="选择的属性ID列表")
    predictions: Dict[str, Any] = Field(..., description="预测结果字典")
    
    class Config:
        json_schema_extra = {
            "example": {
                "smiles": "CCO",
                "selected_properties": ["cyp1a2", "logS", "ames"],
                "predictions": {
                    "cyp1a2": {
                        "name": "CYP1A2-inh",
                        "type": "classification",
                        "probability": 0.75,
                        "symbol": "++",
                        "description": "High"
                    },
                    "logS": {
                        "name": "LogS",
                        "type": "regression",
                        "value": -0.77
                    },
                    "ames": {
                        "name": "Ames",
                        "type": "classification",
                        "probability": 0.23,
                        "symbol": "-",
                        "description": "Moderate Low"
                    }
                }
            }
        }


class ADMETBatchPredictionResult(BaseModel):
    """批量ADMET预测结果"""
    
    total_molecules: int = Field(..., description="总分子数")
    results: List[ADMETPredictionResult] = Field(..., description="预测结果列表")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_molecules": 2,
                "results": [
                    {
                        "smiles": "CCO",
                        "selected_properties": ["cyp1a2", "logS"],
                        "predictions": {
                            "cyp1a2": {
                                "name": "CYP1A2-inh",
                                "type": "classification",
                                "probability": 0.75,
                                "symbol": "++",
                                "description": "High"
                            },
                            "logS": {
                                "name": "LogS",
                                "type": "regression",
                                "value": -0.77
                            }
                        }
                    }
                ]
            }
        }


class ADMETPropertyInfo(BaseModel):
    """ADMET属性信息模型"""
    
    id: str = Field(..., description="属性ID")
    name: str = Field(..., description="属性显示名称")
    task_name: str = Field(..., description="后端任务名称")
    category: str = Field(..., description="属性分类")
    type: str = Field(..., description="任务类型 (classification/regression)")
    description: Optional[str] = Field(None, description="属性描述")


class ADMETPropertiesListResponse(BaseModel):
    """ADMET属性列表响应"""
    
    total: int = Field(..., description="属性总数")
    categories: Dict[str, List[ADMETPropertyInfo]] = Field(
        ..., 
        description="按分类组织的属性列表"
    )


