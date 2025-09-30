from fastapi import APIRouter, HTTPException

from app.models.schemas import DruglikenessRequest, DruglikenessResponse
from app.services.druglikeness_service import DruglikenessService
from app.utils.response import success, fail

router = APIRouter()


@router.post("/evaluate")
async def evaluate_druglikeness(request: DruglikenessRequest):
    """
    评估分子的类药性

    根据用户选择的规则评估SMILES字符串的类药性。
    如果未指定规则，则使用所有五种规则（Lipinski、Ghose、Oprea、Veber、Varma）。
    
    可选规则:
    - Lipinski: 分子量≤500, LogP≤5, 氢键供体≤5, 氢键受体≤10
    - Ghose: 分子量160-480, LogP(-0.4-5.6), 摩尔折射率40-130, 原子数20-70
    - Oprea: 可旋转键≤15, 刚性键≥2, 至少一个环结构
    - Veber: 可旋转键≤10, 极性表面积≤140, 氢键供体≤5, 氢键受体≤10
    - Varma: 分子量300-500, 极性表面积≤150, LogD 0-4, 氢键供体≤5, 氢键受体≤10, 可旋转键≤10
    """
    try:
        service = DruglikenessService()
        metrics, matches, total_score = service.evaluate_druglikeness(
            request.smiles, request.rules
        )

        response_data = {
            "metrics": metrics,
            "matches": matches,
            # "total_score": total_score,  # 暂时注释掉
            "smiles": request.smiles,
        }

        return success(data=response_data, message="药物类药性评估成功")

    except ValueError as e:
        return fail(message=str(e), code=400)
    except Exception as e:
        return fail(message=f"服务器内部错误: {str(e)}", code=500)
