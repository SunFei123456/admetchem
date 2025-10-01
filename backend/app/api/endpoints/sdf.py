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
    selected_items: Optional[str] = Query(default=None)
):
    """
    上传SDF文件，提取SMILES并对每条进行综合评估（类药性规则+分子性质）

    - 评估项目选择：多选，通过查询参数 selected_items 指定，用逗号分隔
    - 可用项目：Lipinski, Ghose, Oprea, Veber, Varma, QED, SAscore, Fsp3, MCE18, NPscore
    - 不传则默认评估 Lipinski 规则
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

        # 解析选择的评估项目
        allowed_items = {
            "Lipinski", "Ghose", "Oprea", "Veber", "Varma", 
            "QED", "SAscore", "Fsp3", "MCE18", "NPscore"
        }
        
        if selected_items is None or selected_items.strip() == "":
            # 默认只评估 Lipinski 规则
            items_to_use = ["Lipinski"]
        else:
            # 解析多选项目
            items_to_use = [item.strip() for item in selected_items.split(',') if item.strip()]
            # 验证所有项目是否有效
            invalid_items = [item for item in items_to_use if item not in allowed_items]
            if invalid_items:
                return fail(
                    message=f"无效的评估项目: {invalid_items}. 可用项目: {sorted(list(allowed_items))}", 
                    code=400
                )

        service = DruglikenessService()
        items = []
        for item in results:
            smiles = item.get('smiles')
            if not smiles:
                continue
            try:
                # 使用新的综合评估方法
                evaluation_result = service.evaluate_comprehensive(smiles, selected_items=items_to_use)
                items.append({
                    'smiles': smiles,
                    'selected_items': evaluation_result['selected_items'],
                    'druglikeness_rules': evaluation_result['druglikeness_rules'],
                    'molecular_properties': evaluation_result['molecular_properties']
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