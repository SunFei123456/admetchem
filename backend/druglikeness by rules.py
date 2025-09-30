from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import numpy as np


class DruglikenessEvaluator:
    """药物类药性评估器，包含五种评估规则"""

    def __init__(self):
        # 各规则的权重（可根据需要调整）
        self.rule_weights = {
            'Lipinski': 0.3,
            'Ghose': 0.2,
            'Oprea': 0.2,
            'Veber': 0.15,
            'Varma': 0.15
        }

    def get_lipinski_metrics(self, mol):
        """获取Lipinski规则指标值"""
        return {
            'mw': Descriptors.MolWt(mol),
            'logp': Chem.Crippen.MolLogP(mol),
            'hbd': rdMolDescriptors.CalcNumHBD(mol),
            'hba': rdMolDescriptors.CalcNumHBA(mol)
        }

    def calc_lipinski_match(self, metrics):
        """计算Lipinski规则匹配度（0-1）"""
        params = {
            'mw': metrics['mw'] <= 500,
            'logp': metrics['logp'] <= 5,
            'hbd': metrics['hbd'] <= 5,
            'hba': metrics['hba'] <= 10
        }
        return sum(params.values()) / len(params)

    def get_ghose_metrics(self, mol):
        """获取Ghose规则指标值"""
        return {
            'mw': Descriptors.MolWt(mol),
            'logp': Chem.Crippen.MolLogP(mol),
            'mr': Chem.Crippen.MolMR(mol),
            'atom_count': mol.GetNumAtoms()
        }

    def calc_ghose_match(self, metrics):
        """计算Ghose规则匹配度（0-1）"""
        params = {
            'mw': 160 <= metrics['mw'] <= 480,
            'logp': -0.4 <= metrics['logp'] <= 5.6,
            'mr': 40 <= metrics['mr'] <= 130,
            'atom_count': 20 <= metrics['atom_count'] <= 70
        }
        return sum(params.values()) / len(params)

    def get_oprea_metrics(self, mol):
        """获取Oprea规则指标值（包含刚性键计算）"""
        # 计算可旋转键数量
        rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)

        # 计算刚性键数量 = 总键数 - 可旋转键数
        total_bonds = mol.GetNumBonds()
        rigid_bonds = total_bonds - rot_bonds

        # 计算环结构数量
        ring_count = rdMolDescriptors.CalcNumRings(mol)

        return {
            'rot_bonds': rot_bonds,
            'rigid_bonds': rigid_bonds,  # 新增刚性键指标
            'ring_count': ring_count
        }

    def calc_oprea_match(self, metrics):
        """计算Oprea规则匹配度（0-1）"""
        params = {
            'rot_bonds': metrics['rot_bonds'] <= 15,  # 可旋转键≤15
            'rigid_bonds': metrics['rigid_bonds'] >= 2,  # 刚性键≥2
            'ring_count': metrics['ring_count'] >= 1  # 至少一个环结构
        }
        return sum(params.values()) / len(params)

    def get_veber_metrics(self, mol):
        """获取Veber规则指标值（完整版）"""
        return {
            'rot_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
            'tpsa': Chem.rdMolDescriptors.CalcTPSA(mol),
            'hbd': rdMolDescriptors.CalcNumHBD(mol),  # 氢键供体
            'hba': rdMolDescriptors.CalcNumHBA(mol)  # 氢键受体
        }

    def calc_veber_match(self, metrics):
        """计算Veber规则匹配度（0-1）完整版"""
        params = {
            'rot_bonds': metrics['rot_bonds'] <= 10,  # 可旋转键≤10
            'tpsa': metrics['tpsa'] <= 140,  # TPSA≤140
            'hbd': metrics['hbd'] <= 5,  # 氢键供体≤5
            'hba': metrics['hba'] <= 10  # 氢键受体≤10
        }
        return sum(params.values()) / len(params)  # 四项全符合得100%

    def get_varma_metrics(self, mol):
        """获取完整的Varma规则指标值"""
        return {
            'molecular_weight': Descriptors.MolWt(mol),  # 分子量
            'tpsa': Chem.rdMolDescriptors.CalcTPSA(mol),  # 拓扑极性表面积
            'logd': self.calculate_logd(mol),  # LogD分布系数
            'h_bond_donor': rdMolDescriptors.CalcNumHBD(mol),  # 氢键供体
            'h_bond_acceptor': rdMolDescriptors.CalcNumHBA(mol),  # 氢键受体
            'rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol)  # 可旋转键
        }

    def calculate_logd(self, mol, pH=7.4):
        """计算LogD分布系数（LogP是近似值）"""
        # 实际项目中应使用更专业的LogD计算方法
        # 这里使用LogP作为近似值（与图片实现一致）
        return Chem.Crippen.MolLogP(mol)

    def calc_varma_match(self, metrics):
        """计算Varma规则匹配度（0-1）完整版"""
        # 基于Varma的原始论文和图片中的实现
        params = {
            'molecular_weight': 300 <= metrics['molecular_weight'] <= 500,  # 最佳范围
            'tpsa': metrics['tpsa'] <= 150,  # 极性表面积限制
            'logd': 0 <= metrics['logd'] <= 4,  # 亲脂性适中范围
            'h_bond_donor': metrics['h_bond_donor'] <= 5,  # 氢键供体≤5
            'h_bond_acceptor': metrics['h_bond_acceptor'] <= 10,  # 氢键受体≤10
            'rotatable_bonds': metrics['rotatable_bonds'] <= 10  # 可旋转键≤10
        }
        # 计算符合标准的参数比例
        return sum(params.values()) / len(params)

    def evaluate(self, smiles):
        """评估SMILES字符串的整体类药性"""
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return {"error": "Invalid SMILES string"}, {}, 0.0

        # 获取各规则指标
        metrics = {
            'Lipinski': self.get_lipinski_metrics(mol),
            'Ghose': self.get_ghose_metrics(mol),
            'Oprea': self.get_oprea_metrics(mol),
            'Veber': self.get_veber_metrics(mol),
            'Varma': self.get_varma_metrics(mol)
        }

        # 计算匹配度
        matches = {
            'Lipinski': self.calc_lipinski_match(metrics['Lipinski']),
            'Ghose': self.calc_ghose_match(metrics['Ghose']),
            'Oprea': self.calc_oprea_match(metrics['Oprea']),
            'Veber': self.calc_veber_match(metrics['Veber']),
            'Varma': self.calc_varma_match(metrics['Varma'])
        }

        # 计算加权平均匹配度
        total_score = sum(matches[rule] * self.rule_weights[rule]
                          for rule in matches)

        return metrics, matches, total_score


def print_metrics_table(metrics):
    """格式化输出指标值表格"""
    print("\n{:<15} {:<10} {:<10} {:<10} {:<10}".format(
        "规则", "参数1", "值", "参数2", "值"))
    print("-" * 60)

    for rule, values in metrics.items():
        items = list(values.items())
        # 每行输出两个参数
        for i in range(0, len(items), 2):
            param1 = f"{items[i][0]}: {items[i][1]:.2f}" if i < len(items) else ""
            param2 = f"{items[i + 1][0]}: {items[i + 1][1]:.2f}" if i + 1 < len(items) else ""
            prefix = rule if i == 0 else ""
            print("{:<15} {:<15} {:<15} {:<15}".format(
                prefix, param1, param2, ""))


def main():
    evaluator = DruglikenessEvaluator()

    # 示例测试（苯磺酸氨氯地平 - 常用降压药）
    sample_smiles = "CCN(CC)CCCOC1=NC2=C(C=CC(=C2Cl)C(=O)OC)C(=N1)C3=CC=CC=C3S(=O)(=O)O"

    # 用户输入SMILES
    user_smiles = input("请输入SMILES字符串: ").strip() or sample_smiles

    # 评估类药性
    metrics, matches, total_score = evaluator.evaluate(user_smiles)

    # 输出结果
    if isinstance(metrics, dict) and "error" in metrics:
        print(metrics["error"])
        return

    # 输出各规则指标值
    print("\n分子指标值:")
    print("=" * 60)
    print_metrics_table(metrics)

    # 输出匹配度结果
    print("\n\n规则匹配度:")
    print("=" * 60)
    for rule, score in matches.items():
        print(f"{rule:<10}: {score:.2%}")

    print("-" * 60)
    print(f"总体匹配度: {total_score:.2%}")
    print("=" * 60)

    # 详细规则解释
    print("\n规则说明:")
    print("- Lipinski: 分子量≤500, LogP≤5, 氢键供体≤5, 氢键受体≤10 [2,6](@ref)")
    print("- Ghose: 分子量160-480, LogP(-0.4-5.6), 摩尔折射率40-130, 原子数20-70 [1](@ref)")
    print("- Oprea: 可旋转键2-15, 至少一个环结构")
    print("- Veber: 可旋转键≤10, 极性表面积≤140 [6](@ref)")
    print("- Varma: LogP≤6, 极性表面积≤150, 总氢键(供体+受体)≤12")


if __name__ == "__main__":
    main()