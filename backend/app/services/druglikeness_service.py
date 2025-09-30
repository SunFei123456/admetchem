from typing import Dict, Any, Tuple
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


class DruglikenessService:
    """药物类药性评估服务"""

    def __init__(self):
        # 各规则的权重（可根据需要调整）
        self.rule_weights = {
            "Lipinski": 0.3,
            "Ghose": 0.2,
            "Oprea": 0.2,
            "Veber": 0.15,
            "Varma": 0.15,
        }

    def get_lipinski_metrics(self, mol: Chem.Mol) -> Dict[str, float]:
        """获取Lipinski规则指标值"""
        return {
            "mw": Descriptors.MolWt(mol),
            "logp": Chem.Crippen.MolLogP(mol),
            "hbd": rdMolDescriptors.CalcNumHBD(mol),
            "hba": rdMolDescriptors.CalcNumHBA(mol),
        }

    def calc_lipinski_match(self, metrics: Dict[str, float]) -> float:
        """计算Lipinski规则匹配度（0-1）"""
        params = {
            "mw": metrics["mw"] <= 500,
            "logp": metrics["logp"] <= 5,
            "hbd": metrics["hbd"] <= 5,
            "hba": metrics["hba"] <= 10,
        }
        return sum(params.values()) / len(params)

    def get_ghose_metrics(self, mol: Chem.Mol) -> Dict[str, float]:
        """获取Ghose规则指标值"""
        return {
            "mw": Descriptors.MolWt(mol),
            "logp": Chem.Crippen.MolLogP(mol),
            "mr": Chem.Crippen.MolMR(mol),
            "atom_count": mol.GetNumAtoms(),
        }

    def calc_ghose_match(self, metrics: Dict[str, float]) -> float:
        """计算Ghose规则匹配度（0-1）"""
        params = {
            "mw": 160 <= metrics["mw"] <= 480,
            "logp": -0.4 <= metrics["logp"] <= 5.6,
            "mr": 40 <= metrics["mr"] <= 130,
            "atom_count": 20 <= metrics["atom_count"] <= 70,
        }
        return sum(params.values()) / len(params)

    def get_oprea_metrics(self, mol: Chem.Mol) -> Dict[str, float]:
        """获取Oprea规则指标值（包含刚性键计算）"""
        # 计算可旋转键数量
        rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)

        # 计算刚性键数量 = 总键数 - 可旋转键数
        total_bonds = mol.GetNumBonds()
        rigid_bonds = total_bonds - rot_bonds

        # 计算环结构数量
        ring_count = rdMolDescriptors.CalcNumRings(mol)

        return {
            "rot_bonds": rot_bonds,
            "rigid_bonds": rigid_bonds,
            "ring_count": ring_count,
        }

    def calc_oprea_match(self, metrics: Dict[str, float]) -> float:
        """计算Oprea规则匹配度（0-1）"""
        params = {
            "rot_bonds": metrics["rot_bonds"] <= 15,
            "rigid_bonds": metrics["rigid_bonds"] >= 2,
            "ring_count": metrics["ring_count"] >= 1,
        }
        return sum(params.values()) / len(params)

    def get_veber_metrics(self, mol: Chem.Mol) -> Dict[str, float]:
        """获取Veber规则指标值（完整版）"""
        return {
            "rot_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "tpsa": Chem.rdMolDescriptors.CalcTPSA(mol),
            "hbd": rdMolDescriptors.CalcNumHBD(mol),
            "hba": rdMolDescriptors.CalcNumHBA(mol),
        }

    def calc_veber_match(self, metrics: Dict[str, float]) -> float:
        """计算Veber规则匹配度（0-1）完整版"""
        params = {
            "rot_bonds": metrics["rot_bonds"] <= 10,
            "tpsa": metrics["tpsa"] <= 140,
            "hbd": metrics["hbd"] <= 5,
            "hba": metrics["hba"] <= 10,
        }
        return sum(params.values()) / len(params)

    def get_varma_metrics(self, mol: Chem.Mol) -> Dict[str, float]:
        """获取完整的Varma规则指标值"""
        return {
            "molecular_weight": Descriptors.MolWt(mol),
            "tpsa": Chem.rdMolDescriptors.CalcTPSA(mol),
            "logd": self.calculate_logd(mol),
            "h_bond_donor": rdMolDescriptors.CalcNumHBD(mol),
            "h_bond_acceptor": rdMolDescriptors.CalcNumHBA(mol),
            "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        }

    def calculate_logd(self, mol: Chem.Mol, pH: float = 7.4) -> float:
        """计算LogD分布系数（LogP是近似值）"""
        # 实际项目中应使用更专业的LogD计算方法
        # 这里使用LogP作为近似值（与图片实现一致）
        return Chem.Crippen.MolLogP(mol)

    def calc_varma_match(self, metrics: Dict[str, float]) -> float:
        """计算Varma规则匹配度（0-1）完整版"""
        # 基于Varma的原始论文和图片中的实现
        params = {
            "molecular_weight": 300 <= metrics["molecular_weight"] <= 500,
            "tpsa": metrics["tpsa"] <= 150,
            "logd": 0 <= metrics["logd"] <= 4,
            "h_bond_donor": metrics["h_bond_donor"] <= 5,
            "h_bond_acceptor": metrics["h_bond_acceptor"] <= 10,
            "rotatable_bonds": metrics["rotatable_bonds"] <= 10,
        }
        # 计算符合标准的参数比例
        return sum(params.values()) / len(params)

    def evaluate_druglikeness(
        self, smiles: str, selected_rules: list = None
    ) -> Tuple[Dict[str, Any], Dict[str, float], float]:
        """评估SMILES字符串的整体类药性
        
        Args:
            smiles: SMILES字符串
            selected_rules: 用户选择的规则列表，如果为None则返回所有规则
        """
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            raise ValueError("Invalid SMILES string")

        # 所有可用的规则
        all_rules = ["Lipinski", "Ghose", "Oprea", "Veber", "Varma"]
        
        # 如果没有指定规则，使用所有规则
        if selected_rules is None:
            rules_to_evaluate = all_rules
        else:
            # 验证用户选择的规则是否有效
            invalid_rules = [rule for rule in selected_rules if rule not in all_rules]
            if invalid_rules:
                raise ValueError(f"Invalid rules: {invalid_rules}. Available rules: {all_rules}")
            rules_to_evaluate = selected_rules

        # 获取各规则指标
        metrics = {}
        matches = {}
        
        for rule in rules_to_evaluate:
            if rule == "Lipinski":
                metrics[rule] = self.get_lipinski_metrics(mol)
                matches[rule] = self.calc_lipinski_match(metrics[rule])
            elif rule == "Ghose":
                metrics[rule] = self.get_ghose_metrics(mol)
                matches[rule] = self.calc_ghose_match(metrics[rule])
            elif rule == "Oprea":
                metrics[rule] = self.get_oprea_metrics(mol)
                matches[rule] = self.calc_oprea_match(metrics[rule])
            elif rule == "Veber":
                metrics[rule] = self.get_veber_metrics(mol)
                matches[rule] = self.calc_veber_match(metrics[rule])
            elif rule == "Varma":
                metrics[rule] = self.get_varma_metrics(mol)
                matches[rule] = self.calc_varma_match(metrics[rule])

        # 计算加权平均匹配度（只针对选择的规则）
        if matches:
            # 重新计算权重，确保选择的规则权重总和为1
            selected_weights = {rule: self.rule_weights[rule] for rule in rules_to_evaluate}
            total_weight = sum(selected_weights.values())
            normalized_weights = {rule: weight/total_weight for rule, weight in selected_weights.items()}
            
            total_score = sum(
                matches[rule] * normalized_weights[rule] for rule in matches
            )
        else:
            total_score = 0.0

        return metrics, matches, total_score
