from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    DruglikenessRequest, 
    DruglikenessResponse,
    ComprehensiveEvaluationResponse
)
from app.services.druglikeness_service import DruglikenessService
from app.utils.response import success, fail

router = APIRouter()


@router.post("/evaluate", response_model=None)
async def evaluate_druglikeness(request: DruglikenessRequest):
    """
    综合评估分子的类药性和分子性质
    
    支持两种模式：
    1. 新版模式（推荐）：使用 selected_items 参数选择要计算的项目
    2. 旧版模式（向后兼容）：使用 rules 参数仅评估类药性规则
    
    可选项目（selected_items）:
    - 类药性规则: Lipinski, Ghose, Oprea, Veber, Varma
    - 分子性质: QED, SAscore, Fsp3, MCE18, NPscore
    
    规则详情:
    - Lipinski: 分子量≤500, LogP≤5, 氢键供体≤5, 氢键受体≤10
    - Ghose: 分子量160-480, LogP(-0.4-5.6), 摩尔折射率40-130, 原子数20-70
    - Oprea: 可旋转键≤15, 刚性键≥2, 至少一个环结构
    - Veber: 可旋转键≤10, 极性表面积≤140, 氢键供体≤5, 氢键受体≤10
    - Varma: 分子量300-500, 极性表面积≤150, LogD 0-4, 氢键供体≤5, 氢键受体≤10, 可旋转键≤10
    
    性质详情:
    - QED: 类药性定量评估 (0-1, >0.67为优秀)
    - SAscore: 合成可及性 (1-10, ≤6为优秀)
    - Fsp3: 碳饱和度 (0-1, ≥0.42为优秀)
    - MCE18: 结构复杂性 (0-100+, ≥45为优秀)
    - NPscore: 天然产物相似性 (-5到5, 正值表示更像天然产物)
    """
    try:
        service = DruglikenessService()
        
        # 判断使用新版还是旧版API
        if request.selected_items is not None:
            # 新版：综合评估
            result = service.evaluate_comprehensive(
                request.smiles, 
                request.selected_items
            )
            
            response_data = {
                "selected_items": result.get("selected_items", []),
                "druglikeness_rules": result.get("druglikeness_rules"),
                "molecular_properties": result.get("molecular_properties"),
                "smiles": request.smiles,
            }
            
            return success(
                data=response_data, 
                message="综合评估成功"
            )
        else:
            # 旧版：仅评估类药性规则（向后兼容）
            metrics, matches, total_score = service.evaluate_druglikeness(
                request.smiles, 
                request.rules
            )

            response_data = {
                "metrics": metrics,
                "matches": matches,
                # "total_score": total_score,  # 暂时注释掉
                "smiles": request.smiles,
            }

            return success(
                data=response_data, 
                message="药物类药性评估成功"
            )

    except ValueError as e:
        return fail(message=str(e), code=400)
    except Exception as e:
        return fail(message=f"服务器内部错误: {str(e)}", code=500)
