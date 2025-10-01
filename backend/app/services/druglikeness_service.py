import os
import gzip
import pickle
import math
from typing import Dict, Any, Tuple, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED


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
        
        # SAscore和NPscore数据缓存
        self._fscores = None
        self._npscores = None

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

    # ========== 分子性质计算方法（SAscore, NPscore, QED, Fsp3, MCE18）==========
    
    def _read_fragment_scores(self) -> None:
        """加载SAscore所需的片段得分数据"""
        if self._fscores is not None:
            return
            
        data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'fpscores.pkl.gz')
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                "请下载fpscores.pkl.gz并放在backend目录: "
                "https://github.com/rdkit/rdkit/tree/master/Contrib/SA_Score"
            )
        self._fscores = pickle.load(gzip.open(data_path, 'rb'))
        self._fscores = dict((x[1], float(x[0])) for x in self._fscores)

    def _num_bridgeheads_and_spiro(self, mol: Chem.Mol) -> Tuple[int, int]:
        """计算桥头原子和螺原子数量"""
        n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        n_bridgeheads = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        return n_bridgeheads, n_spiro

    def calculate_sascore(self, mol: Chem.Mol) -> float:
        """计算合成可及性分数(SAscore)
        
        范围: 1-10，值越低表示越容易合成
        建议阈值: ≤6 为优秀
        """
        self._read_fragment_scores()

        # 片段得分计算
        fp = rdMolDescriptors.GetMorganFingerprint(mol, 2)
        fps = fp.GetNonzeroElements()
        fragment_score = sum(self._fscores.get(bit, -4) * count for bit, count in fps.items())
        fragment_score /= sum(fps.values())

        # 复杂度惩罚项
        n_atoms = mol.GetNumAtoms()
        n_chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        n_bridgeheads, n_spiro = self._num_bridgeheads_and_spiro(mol)
        n_macrocycles = sum(1 for ring in mol.GetRingInfo().AtomRings() if len(ring) > 8)

        penalty = (
                - (n_atoms ** 1.005 - n_atoms)  # 大小惩罚
                - math.log10(n_chiral + 1)  # 手性中心惩罚
                - math.log10(n_spiro + 1)  # 螺原子惩罚
                - math.log10(n_bridgeheads + 1)  # 桥头原子惩罚
                - (math.log10(2) if n_macrocycles > 0 else 0)  # 大环惩罚
        )

        # 最终分数归一化
        raw_score = fragment_score + penalty
        sa_score = 11.0 - (raw_score + 4.0 + 1.0) / (2.5 + 4.0) * 9.0

        # 平滑处理极端值
        if sa_score > 8.0:
            sa_score = 8.0 + math.log(sa_score + 1.0 - 9.0)
        if sa_score > 10.0:
            sa_score = 10.0
        elif sa_score < 1.0:
            sa_score = 1.0

        return max(1.0, min(10.0, sa_score))

    def _read_np_scores(self) -> None:
        """加载NPscore所需的片段得分数据"""
        if self._npscores is not None:
            return
            
        data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'publicnp.model.gz')
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                "请下载publicnp.model.gz并放在backend目录: "
                "https://github.com/rdkit/rdkit/tree/master/Contrib/NP_Score"
            )
        with gzip.open(data_path, 'rb') as f:
            self._npscores = pickle.load(f)

    def calculate_npscore(self, mol: Chem.Mol) -> float:
        """计算天然产物相似性分数(NPscore)
        
        范围: -5到5
        正值表示更像天然产物，负值表示更像合成化合物
        """
        self._read_np_scores()

        # 使用预训练的天然产物模型计算得分
        fp = rdMolDescriptors.GetMorganFingerprint(mol, 2)
        features = fp.GetNonzeroElements()

        # 计算分子指纹特征的总和
        score = 0.0
        for feat_id, count in features.items():
            if feat_id in self._npscores:
                score += self._npscores[feat_id]

        # 归一化处理
        return score / 10

    def calculate_qed(self, mol: Chem.Mol) -> float:
        """计算QED (Quantitative Estimate of Drug-likeness)
        
        范围: 0-1
        建议阈值: >0.67 为优秀
        """
        return QED.qed(mol)

    def calculate_fsp3(self, mol: Chem.Mol) -> float:
        """计算Fsp3 (碳饱和度)
        
        范围: 0-1
        建议阈值: ≥0.42 为优秀
        """
        carbon_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6]
        if not carbon_atoms:
            return 0.0
        sp3_carbons = [atom for atom in carbon_atoms if atom.GetHybridization() == Chem.HybridizationType.SP3]
        return len(sp3_carbons) / len(carbon_atoms)

    def calculate_mce18(self, mol: Chem.Mol) -> float:
        """计算MCE-18 (结构复杂性)
        
        范围: 0-100+
        建议阈值: ≥45 为优秀
        """
        chiral_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        spiro_atoms = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        fsp3 = self.calculate_fsp3(mol)
        
        cyc_sp3 = len([atom for atom in mol.GetAtoms()
                       if atom.IsInRing() and atom.GetHybridization() == Chem.HybridizationType.SP3])
        acyc_sp3 = len([atom for atom in mol.GetAtoms()
                        if not atom.IsInRing() and atom.GetHybridization() == Chem.HybridizationType.SP3])

        mce18 = (int(mol.GetRingInfo().NumRings() > 0) +
                 int(chiral_centers > 0) +
                 int(spiro_atoms > 0) +
                 fsp3 +
                 (cyc_sp3 - acyc_sp3) / (1 + fsp3)) * 100
        return mce18

    def calculate_molecular_properties(self, mol: Chem.Mol) -> Dict[str, Any]:
        """计算所有分子性质并返回字典结果
        
        包括: QED, SAscore, Fsp3, MCE18, NPscore
        """
        results = {}

        # 1. QED计算
        try:
            qed = self.calculate_qed(mol)
            results['QED'] = qed
            results['QED_status'] = 'excellent' if qed > 0.67 else 'poor'
        except Exception as e:
            results['QED'] = None
            results['QED_status'] = 'error'
            results['QED_error'] = str(e)

        # 2. SAscore计算
        try:
            sa_score = self.calculate_sascore(mol)
            results['SAscore'] = sa_score
            results['SAscore_status'] = 'excellent' if sa_score <= 6 else 'poor'
        except Exception as e:
            results['SAscore'] = None
            results['SAscore_status'] = 'error'
            results['SAscore_error'] = str(e)

        # 3. Fsp3计算
        try:
            fsp3 = self.calculate_fsp3(mol)
            results['Fsp3'] = fsp3
            results['Fsp3_status'] = 'excellent' if fsp3 >= 0.42 else 'poor'
        except Exception as e:
            results['Fsp3'] = None
            results['Fsp3_status'] = 'error'
            results['Fsp3_error'] = str(e)

        # 4. MCE-18计算
        try:
            mce18 = self.calculate_mce18(mol)
            results['MCE18'] = mce18
            results['MCE18_status'] = 'excellent' if mce18 >= 45 else 'poor'
        except Exception as e:
            results['MCE18'] = None
            results['MCE18_status'] = 'error'
            results['MCE18_error'] = str(e)

        # 5. NPscore计算
        try:
            np_score = self.calculate_npscore(mol)
            results['NPscore'] = np_score
            results['NPscore_status'] = 'NP-like' if np_score > 0 else 'synthetic-like'
        except Exception as e:
            results['NPscore'] = None
            results['NPscore_status'] = 'error'
            results['NPscore_error'] = str(e)

        return results

    def evaluate_molecular_properties_from_smiles(self, smiles: str) -> Dict[str, Any]:
        """从SMILES字符串评估分子性质
        
        Args:
            smiles: SMILES字符串
            
        Returns:
            包含所有分子性质的字典
        """
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            raise ValueError("Invalid SMILES string")
        
        return self.calculate_molecular_properties(mol)

    def evaluate_comprehensive(self, smiles: str, selected_items: list = None) -> Dict[str, Any]:
        """综合评估：根据用户选择的项目进行计算
        
        支持的项目：
        - 类药性规则: Lipinski, Ghose, Oprea, Veber, Varma
        - 分子性质: QED, SAscore, Fsp3, MCE18, NPscore
        
        Args:
            smiles: SMILES字符串
            selected_items: 用户选择的项目列表，如 ['Lipinski', 'QED', 'SAscore']
                          如果为None，则计算所有项目
        
        Returns:
            {
                'druglikeness_rules': {
                    'metrics': {...},  # 各规则的详细指标
                    'matches': {...},  # 各规则的匹配度
                    'total_score': float  # 加权总分
                },
                'molecular_properties': {
                    'QED': ...,
                    'SAscore': ...,
                    ...
                },
                'selected_items': [...]  # 用户选择的项目
            }
        """
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            raise ValueError("Invalid SMILES string")
        
        # 定义所有可用的项目
        druglikeness_rules = ["Lipinski", "Ghose", "Oprea", "Veber", "Varma"]
        molecular_properties = ["QED", "SAscore", "Fsp3", "MCE18", "NPscore"]
        all_items = druglikeness_rules + molecular_properties
        
        # 如果没有指定，计算所有项目
        if selected_items is None:
            selected_items = all_items
        
        # 验证选择的项目
        invalid_items = [item for item in selected_items if item not in all_items]
        if invalid_items:
            raise ValueError(f"Invalid items: {invalid_items}. Available items: {all_items}")
        
        result = {
            'selected_items': selected_items,
            'druglikeness_rules': {},  # 初始化为空字典
            'molecular_properties': {}  # 初始化为空字典
        }
        
        # 分离类药性规则和分子性质
        selected_rules = [item for item in selected_items if item in druglikeness_rules]
        selected_props = [item for item in selected_items if item in molecular_properties]
        
        # 1. 评估类药性规则（如果有选择）
        if selected_rules:
            metrics, matches, total_score = self.evaluate_druglikeness(smiles, selected_rules)
            result['druglikeness_rules'] = {
                'metrics': metrics,
                'matches': matches,
                'total_score': total_score
            }
        
        # 2. 计算分子性质（如果有选择）
        if selected_props:
            all_props = self.calculate_molecular_properties(mol)
            
            # 只返回用户选择的性质
            filtered_props = {}
            for prop in selected_props:
                if prop in all_props:
                    filtered_props[prop] = all_props[prop]
                    # 同时包含状态
                    status_key = f"{prop}_status"
                    if status_key in all_props:
                        filtered_props[status_key] = all_props[status_key]
                    # 如果有错误信息也包含
                    error_key = f"{prop}_error"
                    if error_key in all_props:
                        filtered_props[error_key] = all_props[error_key]
            
            result['molecular_properties'] = filtered_props
        
        return result
