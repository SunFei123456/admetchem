from rdkit import Chem
from rdkit.Chem import PandasTools
import pandas as pd
import argparse


def sdf_to_smiles(input_sdf, output_file=None):
    """
    将SDF文件转换为SMILES列表
    :param input_sdf: 输入的SDF文件路径
    :param output_file: 输出文件路径（可选）
    :return: 包含分子名称和SMILES的DataFrame
    """
    # 1. 读取SDF文件
    sdf_supplier = Chem.SDMolSupplier(input_sdf)

    # 2. 准备数据容器
    results = {
        'original_name': [],
        'smiles': []
    }

    # 3. 处理每个分子
    for idx, mol in enumerate(sdf_supplier):
        if mol is None:
            print(f"⚠️ 警告：跳过无法解析的分子 #{idx + 1}")
            continue

        # 获取分子名称
        if mol.HasProp("_Name"):
            name = mol.GetProp("_Name")
        else:
            name = f"Compound_{idx + 1}"

        # 转换SMILES（标准化处理）
        smiles = Chem.MolToSmiles(mol,
                                  isomericSmiles=True,  # 保留立体化学信息
                                  kekuleSmiles=True,  # 凯库勒式表达
                                  canonical=True  # 标准化SMILES
                                  )

        results['original_name'].append(name)
        results['smiles'].append(smiles)

    # 4. 创建DataFrame
    df = pd.DataFrame(results)

    # 5. 处理输出
    if output_file:
        if output_file.endswith('.csv'):
            df.to_csv(output_file, index=False)
        elif output_file.endswith('.xlsx'):
            df.to_excel(output_file, index=False)
        else:
            with open(output_file, 'w') as f:
                for _, row in df.iterrows():
                    f.write(f"{row['original_name']}\t{row['smiles']}\n")

        print(f"✅ 转换完成: {len(df)}个分子已保存至 {output_file}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SDF转SMILES转换器')

    parser.add_argument('-o', '--output', default='output.smi', help='输出文件路径')
    args = parser.parse_args()
    input = "input.sdf"
    # 执行转换
    result_df = sdf_to_smiles(input, args.output)

    # 打印前5个分子
    if not result_df.empty:
        print("\n示例结果:")
        print(result_df.head(20).to_string(index=False))