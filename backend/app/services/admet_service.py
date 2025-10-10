"""
ADMET预测服务
封装predict.py的功能，提供统一的服务接口
"""
import sys
import os
from typing import List, Dict, Any, Union

# 添加admet-prediction路径到sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(os.path.dirname(current_dir))
admet_dir = os.path.join(backend_dir, 'admet-prediction', 'train', 'model1')
sys.path.insert(0, admet_dir)

# 导入预测函数
from predict import predict_smiles, dic


class ADMETService:
    """ADMET预测服务类"""
    
    # 前端属性ID到后端任务名称的映射
    PROPERTY_MAPPING = {
        # Biophysics - 生物物理学性质
        'cyp1a2': 'CYP1A2-inh',
        'cyp2c9': 'CYP2C9-inh',
        'cyp2c9-sub': 'CYP2C9-sub',
        'cyp2c19': 'CYP2C19-inh',
        'cyp2d6': 'CYP2D6-inh',
        'cyp2d6-sub': 'CYP2D6-sub',
        'cyp3a4': 'CYP3A4-inh',
        'cyp3a4-sub': 'CYP3A4-sub',
        'herg': 'hERG',
        'pgp': 'Pgp-inh',
        
        # Physical Chemistry - 物理化学性质
        'logS': 'LogS',
        'logP': 'LogP',
        'logD': 'LogD',
        'hydration': 'Hydration Free Energy',
        'pampa': 'PAMPA',
        
        # Physiology - 生理学性质
        'ames': 'Ames',
        'bbb': 'BBB',
        'bioavailability': 'Bioavailability',
        'caco2': 'Caco-2',
        'clearance': 'CL',
        'dili': 'DILI',
        'halflife': 'Drug Half-Life',
        'hia': 'HIA',
        'ld50': 'LD50',
        'ppbr': 'PPBR',
        'skinSen': 'SkinSen',
        'nr-ar-lbd': 'NR-AR-LBD',
        'nr-ar': 'NR-AR',
        'nr-ahr': 'NR-AhR',
        'nr-aromatase': 'NR-Aromatase',
        'nr-er': 'NR-ER',
        'nr-er-lbd': 'NR-ER-LBD',
        'nr-ppar-gamma': 'NR-PPAR-gamma',
        'sr-are': 'SR-ARE',
        'sr-atad5': 'SR-ATAD5',
        'sr-hse': 'SR-HSE',
        'sr-mmp': 'SR-MMP',
        'sr-p53': 'SR-p53',
        'vdss': 'VDss'
    }
    
    def __init__(self):
        """初始化ADMET服务"""
        # 保存model1目录路径，用于切换工作目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        backend_dir = os.path.dirname(os.path.dirname(current_dir))
        self.model_dir = os.path.join(backend_dir, 'admet-prediction', 'train', 'model1')
    
    def validate_smiles(self, smiles: str) -> None:
        """
        验证SMILES字符串
        
        Args:
            smiles: SMILES字符串
            
        Raises:
            ValueError: 如果SMILES无效
        """
        if not smiles or not isinstance(smiles, str):
            raise ValueError("SMILES字符串不能为空")
        
        # 去除首尾空格
        smiles = smiles.strip()
        
        if len(smiles) == 0:
            raise ValueError("SMILES字符串不能为空")
    
    def validate_properties(self, property_ids: List[str]) -> List[str]:
        """
        验证并转换属性ID为后端任务名称
        
        Args:
            property_ids: 前端属性ID列表
            
        Returns:
            task_names: 后端任务名称列表
            
        Raises:
            ValueError: 如果属性ID无效
        """
        if not property_ids or len(property_ids) == 0:
            raise ValueError("必须选择至少一个ADMET属性")
        
        task_names = []
        invalid_ids = []
        
        for prop_id in property_ids:
            if prop_id in self.PROPERTY_MAPPING:
                task_name = self.PROPERTY_MAPPING[prop_id]
                # 验证任务名称是否在支持的任务列表中
                if task_name in dic:
                    task_names.append(task_name)
                else:
                    invalid_ids.append(prop_id)
            else:
                invalid_ids.append(prop_id)
        
        if invalid_ids:
            raise ValueError(f"无效的属性ID: {', '.join(invalid_ids)}")
        
        if len(task_names) == 0:
            raise ValueError("没有有效的ADMET属性")
        
        return task_names
    
    def format_prediction_result(
        self, 
        raw_result: Dict[str, Dict[str, Any]], 
        smiles: str,
        selected_property_ids: List[str]
    ) -> Dict[str, Any]:
        """
        格式化预测结果
        
        Args:
            raw_result: predict_smiles返回的原始结果
            smiles: SMILES字符串
            selected_property_ids: 用户选择的属性ID列表
            
        Returns:
            formatted_result: 格式化后的结果
        """
        # 获取该SMILES的预测结果
        smiles_result = raw_result.get(smiles, {})
        
        # 格式化结果
        formatted_predictions = {}
        
        for prop_id in selected_property_ids:
            task_name = self.PROPERTY_MAPPING.get(prop_id)
            if task_name and task_name in smiles_result:
                task_result = smiles_result[task_name]
                
                # 根据任务类型格式化结果
                task_type = dic.get(task_name)
                
                if task_type == "class" or task_type == "bio" or task_type == "pama":
                    # 分类任务：包含prediction和symbol
                    formatted_predictions[prop_id] = {
                        "name": task_name,
                        "type": "classification",
                        "probability": float(task_result["prediction"]),
                        "symbol": task_result["symbol"],
                        "description": self._get_symbol_description(task_result["symbol"])
                    }
                elif task_type == "reg" or task_type == "hfe":
                    # 回归任务：只有prediction值
                    formatted_predictions[prop_id] = {
                        "name": task_name,
                        "type": "regression",
                        "value": float(task_result["prediction"])
                    }
        
        return {
            "smiles": smiles,
            "selected_properties": selected_property_ids,
            "predictions": formatted_predictions
        }
    
    def _get_symbol_description(self, symbol: str) -> str:
        """
        获取符号的描述
        
        Args:
            symbol: ADMET符号 (---, --, -, +, ++, +++)
            
        Returns:
            description: 符号描述
        """
        descriptions = {
            '---': 'Very Low',
            '--': 'Low',
            '-': 'Moderate Low',
            '+': 'Moderate High',
            '++': 'High',
            '+++': 'Very High'
        }
        return descriptions.get(symbol, 'Unknown')
    
    def predict_single(
        self, 
        smiles: str, 
        property_ids: List[str]
    ) -> Dict[str, Any]:
        """
        预测单个分子的ADMET性质
        
        Args:
            smiles: SMILES字符串
            property_ids: 前端属性ID列表
            
        Returns:
            result: 预测结果
            
        Raises:
            ValueError: 如果输入无效
            Exception: 如果预测失败
        """
        # 验证输入
        self.validate_smiles(smiles)
        task_names = self.validate_properties(property_ids)
        
        # 去除空格
        smiles = smiles.strip()
        
        try:
            # 保存当前工作目录
            original_cwd = os.getcwd()
            
            try:
                # 切换到model1目录（predict.py需要在此目录下运行）
                os.chdir(self.model_dir)
                
                # 调用预测函数
                raw_result = predict_smiles([smiles], task_names)
                
                # 格式化结果
                formatted_result = self.format_prediction_result(
                    raw_result, 
                    smiles, 
                    property_ids
                )
                
                return formatted_result
            finally:
                # 恢复原工作目录
                os.chdir(original_cwd)
            
        except Exception as e:
            raise Exception(f"ADMET预测失败: {str(e)}")
    
    def predict_batch(
        self, 
        smiles_list: List[str], 
        property_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """
        批量预测多个分子的ADMET性质
        
        Args:
            smiles_list: SMILES字符串列表
            property_ids: 前端属性ID列表
            
        Returns:
            results: 预测结果列表
            
        Raises:
            ValueError: 如果输入无效
            Exception: 如果预测失败
        """
        # 验证输入
        if not smiles_list or len(smiles_list) == 0:
            raise ValueError("SMILES列表不能为空")
        
        task_names = self.validate_properties(property_ids)
        
        # 去除空格并验证每个SMILES
        cleaned_smiles = []
        for smiles in smiles_list:
            self.validate_smiles(smiles)
            cleaned_smiles.append(smiles.strip())
        
        try:
            # 保存当前工作目录
            original_cwd = os.getcwd()
            
            try:
                # 切换到model1目录（predict.py需要在此目录下运行）
                os.chdir(self.model_dir)
                
                # 调用预测函数
                raw_result = predict_smiles(cleaned_smiles, task_names)
                
                # 格式化每个分子的结果
                formatted_results = []
                for smiles in cleaned_smiles:
                    formatted_result = self.format_prediction_result(
                        raw_result, 
                        smiles, 
                        property_ids
                    )
                    formatted_results.append(formatted_result)
                
                return formatted_results
            finally:
                # 恢复原工作目录
                os.chdir(original_cwd)
            
        except Exception as e:
            raise Exception(f"批量ADMET预测失败: {str(e)}")


