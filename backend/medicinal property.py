import os
import sys
import gzip
import pickle
import math
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

# ========== SAscore模块内嵌实现 ==========
_fscores = None


def readFragmentScores():
    """加载SAscore所需的片段得分数据"""
    global _fscores
    data_path = os.path.join(os.path.dirname(__file__), 'fpscores.pkl.gz')
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            "请下载fpscores.pkl.gz并放在脚本目录: "
            "https://github.com/rdkit/rdkit/tree/master/Contrib/SA_Score"
        )
    _fscores = pickle.load(gzip.open(data_path, 'rb'))
    _fscores = dict((x[1], float(x[0])) for x in _fscores)


def numBridgeheadsAndSpiro(mol):
    """计算桥头原子和螺原子数量"""
    n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    n_bridgeheads = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return n_bridgeheads, n_spiro


def calculateSAscore(mol):
    """计算合成可及性分数(SAscore)"""
    global _fscores
    if _fscores is None:
        readFragmentScores()

    # 片段得分计算
    fp = rdMolDescriptors.GetMorganFingerprint(mol, 2)
    fps = fp.GetNonzeroElements()
    fragment_score = sum(_fscores.get(bit, -4) * count for bit, count in fps.items())
    fragment_score /= sum(fps.values())

    # 复杂度惩罚项
    n_atoms = mol.GetNumAtoms()
    n_chiral = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    n_bridgeheads, n_spiro = numBridgeheadsAndSpiro(mol)
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


# ========== NPscore模块内嵌实现 ==========
_npscores = None


def readNPScores():
    """加载NPscore所需的片段得分数据"""
    global _npscores
    data_path = os.path.join(os.path.dirname(__file__), 'publicnp.model.gz')
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            "请下载publicnp.model.gz并放在脚本目录: "
            "https://github.com/rdkit/rdkit/tree/master/Contrib/NP_Score"
        )
    with gzip.open(data_path, 'rb') as f:
        _npscores = pickle.load(f)


def calculateNPscore(mol):
    """计算天然产物相似性分数(NPscore)"""
    global _npscores
    if _npscores is None:
        readNPScores()

    # 使用预训练的天然产物模型计算得分
    fp = rdMolDescriptors.GetMorganFingerprint(mol, 2)
    features = fp.GetNonzeroElements()

    # 计算分子指纹特征的总和
    score = 0.0
    for feat_id, count in features.items():
        if feat_id in _npscores:
            score += _npscores[feat_id]

    # 归一化处理
    return score / 10  # 归一化到-5到5的范围


def calculate_molecular_properties(mol):
    """计算所有分子性质并返回字典结果"""
    results = {}

    # 1. QED计算
    try:
        qed = QED.qed(mol)
        results['QED'] = qed
        results['QED_status'] = 'excellent' if qed > 0.67 else 'poor'
    except:
        results['QED'] = None
        results['QED_status'] = 'error'

    # 2. SAscore计算
    try:
        sa_score = calculateSAscore(mol)
        results['SAscore'] = sa_score
        results['SAscore_status'] = 'excellent' if sa_score <= 6 else 'poor'
    except:
        results['SAscore'] = None
        results['SAscore_status'] = 'error'

    # 3. Fsp3计算
    try:
        carbon_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6]
        sp3_carbons = [atom for atom in carbon_atoms if atom.GetHybridization() == Chem.HybridizationType.SP3]
        fsp3 = len(sp3_carbons) / len(carbon_atoms) if carbon_atoms else 0
        results['Fsp3'] = fsp3
        results['Fsp3_status'] = 'excellent' if fsp3 >= 0.42 else 'poor'
    except:
        results['Fsp3'] = None
        results['Fsp3_status'] = 'error'

    # 4. MCE-18计算
    try:
        chiral_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        spiro_atoms = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        sp3 = results.get('Fsp3', 0)
        cyc_sp3 = len([atom for atom in mol.GetAtoms()
                       if atom.IsInRing() and atom.GetHybridization() == Chem.HybridizationType.SP3])
        acyc_sp3 = len([atom for atom in mol.GetAtoms()
                        if not atom.IsInRing() and atom.GetHybridization() == Chem.HybridizationType.SP3])

        mce18 = (int(mol.GetRingInfo().NumRings() > 0) +
                 int(chiral_centers > 0) +
                 int(spiro_atoms > 0) +
                 sp3 +
                 (cyc_sp3 - acyc_sp3) / (1 + sp3)) * 100
        results['MCE18'] = mce18
        results['MCE18_status'] = 'excellent' if mce18 >= 45 else 'poor'
    except:
        results['MCE18'] = None
        results['MCE18_status'] = 'error'

    # 5. NPscore计算
    try:
        np_score = calculateNPscore(mol)
        results['NPscore'] = np_score
        # NPscore越高表示更像天然产物，但无明确阈值
        results['NPscore_status'] = 'NP-like' if np_score > 0 else 'synthetic-like'
    except Exception as e:
        print(f"NPscore计算失败: {str(e)}")
        results['NPscore'] = None
        results['NPscore_status'] = 'error'



    return results


def print_results(results):
    """简洁打印核心性质结果"""
    print("\n分子性质评估结果:")
    print("=" * 60)

    # 核心性质
    core_props = [
        ('QED', 'QED (类药性定量评估)'),
        ('SAscore', 'SAscore (合成可及性)'),
        ('Fsp3', 'Fsp3 (碳饱和度)'),
        ('MCE18', 'MCE18 (结构复杂性)'),
        ('NPscore', 'NPscore (天然产物相似性)')
    ]

    for prop, name in core_props:
        value = results.get(prop)
        status = results.get(f"{prop}_status", "N/A")
        value_str = f"{value:.3f}" if isinstance(value, float) else str(value)
        print(f"{name}: {value_str} [{status}]")

    # PAINS警示结构
    pains_matches = results.get('PAINS', [])
    pains_status = results.get('PAINS_status', 'N/A')
    print(f"\nPAINS警示结构: {pains_status}")
    if pains_matches and pains_matches != ['error']:
        print("匹配的子结构:")
        for i, match in enumerate(pains_matches, 1):
            print(f"  {i}. {match}")

    print("=" * 60)


# 示例使用
if __name__ == "__main__":
    # 从SMILES创建分子
    smiles = "CC(C)OC(=O)CC(=O)CSC1=C(C=C2CCCC2=N1)C#N"  # 您的分子
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("分子创建失败，请检查SMILES格式")
        sys.exit(1)

    # 计算所有性质
    results = calculate_molecular_properties(mol)

    # 打印结果
    print_results(results)