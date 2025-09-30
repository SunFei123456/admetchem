from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from typing import Optional

from app.models.schemas import SdfToSmilesResponse
from app.services.sdf_service import SdfService
from app.services.druglikeness_service import DruglikenessService
from app.utils.response import success, fail

router = APIRouter()


@router.post("/upload", response_model=dict)
async def convert_sdf_file_to_smiles(
    file: UploadFile = File(...),
    isomeric_smiles: bool = True,
    kekule_smiles: bool = True,
    canonical: bool = True
):
    """
    上传SDF文件并转换为SMILES格式
    
    支持直接上传SDF文件进行转换，无需手动输入文件内容。
    转换参数通过查询参数传递。
    """
    try:
        # 检查文件类型
        if not file.filename.lower().endswith('.sdf'):
            return fail(message="请上传SDF格式文件", code=400)
        
        # 读取文件内容
        sdf_content = await file.read()
        sdf_content = sdf_content.decode('utf-8')
        
        service = SdfService()
        
        # 验证SDF内容
        if not service.validate_sdf_content(sdf_content):
            return fail(message="无效的SDF文件内容", code=400)
        
        # 执行转换
        results, total_molecules, successful_conversions, failed_conversions = service.sdf_to_smiles(
            sdf_content=sdf_content,
            isomeric_smiles=isomeric_smiles,
            kekule_smiles=kekule_smiles,
            canonical=canonical
        )
        
        if total_molecules == 0:
            return fail(message="SDF文件中未找到有效分子", code=400)
        
        response_data = {
            "results": results,
            "filename": file.filename
        }
        
        # 固定提示语：仅提示提取成功
        return success(data=response_data, message="提取成功")
        
    except UnicodeDecodeError:
        return fail(message="文件编码错误，请确保文件为UTF-8编码", code=400)
    except ValueError as e:
        return fail(message=str(e), code=400)
    except Exception as e:
        return fail(message=f"服务器内部错误: {str(e)}", code=500)


@router.post("/analyze", response_model=dict)
async def analyze_sdf_druglikeness(
    file: UploadFile = File(...),
    isomeric_smiles: bool = True,
    kekule_smiles: bool = True,
    canonical: bool = True,
    selected_rule: Optional[str] = Query(default=None)
):
    """
    上传SDF文件，提取SMILES并对每条进行药物类药性评估

    - 规则选择：单选，通过查询参数 selected_rule=RuleName 指定；不传则默认 'Lipinski'
    """
    try:
        if not file.filename.lower().endswith('.sdf'):
            return fail(message="请上传SDF格式文件", code=400)

        sdf_content = await file.read()
        sdf_content = sdf_content.decode('utf-8')

        sdf_service = SdfService()
        if not sdf_service.validate_sdf_content(sdf_content):
            return fail(message="无效的SDF文件内容", code=400)

        # 提取SMILES
        results, _, _, _ = sdf_service.sdf_to_smiles(
            sdf_content=sdf_content,
            isomeric_smiles=isomeric_smiles,
            kekule_smiles=kekule_smiles,
            canonical=canonical
        )

        if not results:
            return fail(message="SDF文件中未找到可评估分子", code=400)

        # 规则默认值：仅 Lipinski；只允许单选
        allowed_rules = {"Lipinski", "Ghose", "Oprea", "Veber", "Varma"}
        if selected_rule is None or selected_rule.strip() == "":
            rules_to_use = ["Lipinski"]
        else:
            if selected_rule not in allowed_rules:
                return fail(message=f"无效的规则: {selected_rule}. 可用规则: {sorted(list(allowed_rules))}", code=400)
            rules_to_use = [selected_rule]

        service = DruglikenessService()
        items = []
        for item in results:
            smiles = item.get('smiles')
            if not smiles:
                continue
            try:
                metrics, matches, total_score = service.evaluate_druglikeness(smiles, selected_rules=rules_to_use)
                items.append({
                    'smiles': smiles,
                    'metrics': metrics,
                    'matches': matches,
                    'total_score': total_score
                })
            except Exception:
                # 单个失败不影响整体，静默跳过
                continue

        if not items:
            return fail(message="SDF文件中未找到可评估分子", code=400)

        response_data = {
            'filename': file.filename,
            'items': items
        }

        return success(data=response_data, message="提取成功")

    except UnicodeDecodeError:
        return fail(message="文件编码错误，请确保文件为UTF-8编码", code=400)
    except ValueError as e:
        return fail(message=str(e), code=400)
    except Exception as e:
        return fail(message=f"服务器内部错误: {str(e)}", code=500)