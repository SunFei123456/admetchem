from rdkit import Chem
import io
from typing import List, Dict, Tuple


class SdfService:
    """SDF转SMILES服务类"""
    
    def __init__(self):
        pass
    
    def _preprocess_sdf_content(self, sdf_content: str) -> str:
        """
        预处理SDF内容，确保格式正确
        
        Args:
            sdf_content: 原始SDF内容
            
        Returns:
            str: 处理后的SDF内容
        """
        # 保持原始内容结构，不插入额外行；仅移除潜在的UTF-8 BOM，避免破坏SDF块头三行结构
        if not sdf_content:
            return sdf_content
        return sdf_content.lstrip("\ufeff")
    
    def sdf_to_smiles(self, sdf_content: str, isomeric_smiles: bool = True, 
                     kekule_smiles: bool = True, canonical: bool = True) -> Tuple[List[Dict[str, str]], int, int, int]:
        """
        将SDF内容转换为SMILES列表
        
        Args:
            sdf_content: SDF文件内容字符串
            isomeric_smiles: 是否保留立体化学信息
            kekule_smiles: 是否使用凯库勒式表达
            canonical: 是否标准化SMILES
            
        Returns:
            Tuple[List[Dict[str, str]], int, int, int]: 
            (结果列表, 总分子数, 成功转换数, 失败转换数)
        """
        # 预处理SDF内容，确保格式正确
        processed_content = self._preprocess_sdf_content(sdf_content)
        
        try:
            # 使用BytesIO和ForwardSDMolSupplier
            from io import BytesIO
            sdf_bytes = processed_content.encode('utf-8')
            sdf_io = BytesIO(sdf_bytes)
            sdf_supplier = Chem.ForwardSDMolSupplier(sdf_io)
        except Exception as e:
            raise ValueError(f"无法解析SDF文件: {str(e)}")
        
        results = []
        total_molecules = 0
        successful_conversions = 0
        failed_conversions = 0
        
        # 遍历所有分子
        for idx, mol in enumerate(sdf_supplier):
            total_molecules += 1
            
            if mol is None:
                failed_conversions += 1
                continue
            
            try:
                # 转换SMILES
                smiles = Chem.MolToSmiles(
                    mol,
                    isomericSmiles=isomeric_smiles,
                    kekuleSmiles=kekule_smiles,
                    canonical=canonical
                )
                
                results.append({'smiles': smiles})
                
                successful_conversions += 1
                
            except Exception as e:
                failed_conversions += 1
                continue
        
        return results, total_molecules, successful_conversions, failed_conversions
    
    def validate_sdf_content(self, sdf_content: str) -> bool:
        """
        验证SDF内容是否有效
        
        Args:
            sdf_content: SDF文件内容字符串
            
        Returns:
            bool: 是否有效
        """
        if not sdf_content or not sdf_content.strip():
            return False
        
        try:
            # 预处理SDF内容
            processed_content = self._preprocess_sdf_content(sdf_content)
            
            # 使用BytesIO和ForwardSDMolSupplier
            from io import BytesIO
            sdf_bytes = processed_content.encode('utf-8')
            sdf_io = BytesIO(sdf_bytes)
            supplier = Chem.ForwardSDMolSupplier(sdf_io)
            
            # 检查是否至少有一个有效分子
            for mol in supplier:
                if mol is not None:
                    return True
            
            return False
                    
        except Exception as e:
            print(f"SDF验证错误: {e}")
            return False
