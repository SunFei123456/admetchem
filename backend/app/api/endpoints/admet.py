"""
ADMET预测API端点
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import List, Optional
from app.services.admet_service import ADMETService
from app.utils.response import success, fail
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# 初始化ADMET服务
admet_service = ADMETService()


@router.post("/predict")
async def predict_admet(
    smiles: str = Form(...),
    property_ids: str = Form(...)  # 逗号分隔的属性ID字符串
):
    """
    单个SMILES的ADMET预测
    
    Args:
        smiles: SMILES字符串
        property_ids: 属性ID列表，逗号分隔
        
    Returns:
        预测结果
    """
    try:
        # 解析属性ID列表
        props = [p.strip() for p in property_ids.split(',') if p.strip()]
        
        if not props:
            return fail(message="必须选择至少一个ADMET属性", code=400)
        
        # 调用服务进行预测
        result = admet_service.predict_single(smiles, props)
        
        return success(data=result, message="ADMET预测成功")
        
    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        return fail(message=str(e), code=400)
    except Exception as e:
        logger.error(f"ADMET prediction error: {str(e)}")
        return fail(message=f"ADMET预测失败: {str(e)}", code=500)


@router.post("/predict-batch")
async def predict_admet_batch(
    smiles_list: str = Form(...),  # 换行符分隔的SMILES列表
    property_ids: str = Form(...)
):
    """
    批量SMILES的ADMET预测
    
    Args:
        smiles_list: SMILES列表，换行符分隔
        property_ids: 属性ID列表，逗号分隔
        
    Returns:
        批量预测结果
    """
    try:
        # 解析SMILES列表
        smiles = [s.strip() for s in smiles_list.split('\n') if s.strip()]
        
        if not smiles:
            return fail(message="SMILES列表不能为空", code=400)
        
        # 解析属性ID列表
        props = [p.strip() for p in property_ids.split(',') if p.strip()]
        
        if not props:
            return fail(message="必须选择至少一个ADMET属性", code=400)
        
        # 调用服务进行批量预测
        results = admet_service.predict_batch(smiles, props)
        
        return success(
            data={"results": results, "count": len(results)},
            message=f"成功预测 {len(results)} 个分子"
        )
        
    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        return fail(message=str(e), code=400)
    except Exception as e:
        logger.error(f"Batch ADMET prediction error: {str(e)}")
        return fail(message=f"批量ADMET预测失败: {str(e)}", code=500)


@router.post("/analyze-sdf")
async def analyze_sdf_file(
    file: UploadFile = File(...),
    property_ids: str = Form(...)
):
    """
    上传SDF文件进行ADMET预测
    
    Args:
        file: SDF文件
        property_ids: 属性ID列表，逗号分隔
        
    Returns:
        批量预测结果
    """
    try:
        # 验证文件类型
        if not file.filename.endswith('.sdf'):
            return fail(message="只支持SDF格式文件", code=400)
        
        # 解析属性ID列表
        props = [p.strip() for p in property_ids.split(',') if p.strip()]
        
        if not props:
            return fail(message="必须选择至少一个ADMET属性", code=400)
        
        # 读取文件内容
        content = await file.read()
        
        # 解析SDF文件，提取SMILES
        from rdkit import Chem
        from io import BytesIO
        
        # 将bytes转换为字符串
        sdf_text = content.decode('utf-8')
        
        # 使用RDKit解析SDF
        supplier = Chem.SDMolSupplier()
        supplier.SetData(sdf_text)
        
        smiles_list = []
        for mol in supplier:
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)
                smiles_list.append(smiles)
        
        if not smiles_list:
            return fail(message="无法从SDF文件中提取有效的分子结构", code=400)
        
        # 调用服务进行批量预测
        results = admet_service.predict_batch(smiles_list, props)
        
        return success(
            data={"results": results, "count": len(results)},
            message=f"成功分析 {len(results)} 个分子"
        )
        
    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        return fail(message=str(e), code=400)
    except Exception as e:
        logger.error(f"SDF analysis error: {str(e)}")
        return fail(message=f"SDF文件分析失败: {str(e)}", code=500)


@router.get("/properties")
async def get_admet_properties():
    """
    获取所有支持的ADMET属性列表
    
    Returns:
        属性列表及分类
    """
    try:
        properties = {
            "biophysics": [
                {"id": "cyp1a2", "name": "CYP1A2-inh"},
                {"id": "cyp2c9", "name": "CYP2C9-inh"},
                {"id": "cyp2c9-sub", "name": "CYP2C9-sub"},
                {"id": "cyp2c19", "name": "CYP2C19-inh"},
                {"id": "cyp2d6", "name": "CYP2D6-inh"},
                {"id": "cyp2d6-sub", "name": "CYP2D6-sub"},
                {"id": "cyp3a4", "name": "CYP3A4-inh"},
                {"id": "cyp3a4-sub", "name": "CYP3A4-sub"},
                {"id": "herg", "name": "hERG"},
                {"id": "pgp", "name": "Pgp-inh"}
            ],
            "physical_chemistry": [
                {"id": "logS", "name": "LogS"},
                {"id": "logP", "name": "LogP"},
                {"id": "logD", "name": "LogD"},
                {"id": "hydration", "name": "Hydration Free Energy"},
                {"id": "pampa", "name": "PAMPA"}
            ],
            "physiology": [
                {"id": "ames", "name": "Ames"},
                {"id": "bbb", "name": "BBB"},
                {"id": "bioavailability", "name": "Bioavailability"},
                {"id": "caco2", "name": "Caco-2"},
                {"id": "clearance", "name": "CL"},
                {"id": "dili", "name": "DILI"},
                {"id": "halflife", "name": "Drug Half-Life"},
                {"id": "hia", "name": "HIA"},
                {"id": "ld50", "name": "LD50"},
                {"id": "ppbr", "name": "PPBR"},
                {"id": "skinSen", "name": "SkinSen"},
                {"id": "nr-ar-lbd", "name": "NR-AR-LBD"},
                {"id": "nr-ar", "name": "NR-AR"},
                {"id": "nr-ahr", "name": "NR-AhR"},
                {"id": "nr-aromatase", "name": "NR-Aromatase"},
                {"id": "nr-er", "name": "NR-ER"},
                {"id": "nr-er-lbd", "name": "NR-ER-LBD"},
                {"id": "nr-ppar-gamma", "name": "NR-PPAR-gamma"},
                {"id": "sr-are", "name": "SR-ARE"},
                {"id": "sr-atad5", "name": "SR-ATAD5"},
                {"id": "sr-hse", "name": "SR-HSE"},
                {"id": "sr-mmp", "name": "SR-MMP"},
                {"id": "sr-p53", "name": "SR-p53"},
                {"id": "vdss", "name": "VDss"}
            ]
        }
        
        return success(data=properties, message="获取ADMET属性列表成功")
        
    except Exception as e:
        logger.error(f"Get properties error: {str(e)}")
        return fail(message=f"获取属性列表失败: {str(e)}", code=500)


