
# 用于保存某个分子的相关信息。
from argparse import Namespace
from typing import List

import numpy as np
from rdkit import Chem

from features import get_features_generator


class MoleculeDatapoint:
    """
            初始化一个 MoleculeDatapoint，它包含一个分子。

            :param line:逗号分隔生成的字符串列表，表示数据 CSV 文件中的一行
            :param args: 参数
            :param features: 一个 NumPy 数组，包含额外的特征（例如 Morgan 指纹）
            :param use_compound_names: 是否在数据 CSV 文件的每一行中包含化合物名称
    """
    # init该类中主要保存了这几个属性：
    # 假如说一个line 如下 ["Ethanol", "CCO", "0.8", "1.2"] （一个line就是csv文件中的一行）
    # self.compound_name = "Ethanol" 这个是CCO 的名称，也就是乙醇。如果init构造函数的use_compound_names为true才会保存
    # self.smiles = "CCO"
    # self.mol = Chem.MolFromSmiles("CCO") 该分子的mol对象
    # self.targets = ["0.8","1.2"] 0.8表示毒性 1.2表示溶解度
    # self.features = [0.5,1.5....] 表示该分子的embedding。可能是一个分子指纹，也可能是预训练的结果
    def __init__(self,
                 line: List[str],
                 args: Namespace = None,
                 features: np.ndarray = None,
                 use_compound_names: bool = False):

        if args is not None:
            self.features_generator = args["features_generator"]
            self.args = args
        else:
            self.features_generator = self.args = None

        # 如果self.features_generator不为空，那么就是用分子指纹来代表当前分子的embedding
        # 如果self.features不为空，那么就用features来代表当前分子的embedding。（用预训练的结果 代表当前分子的embedding）
        # 也就是说，self.features_generator和self.features不能同时存在
        if features is not None and self.features_generator is not None:
            raise ValueError('Currently cannot provide both loaded features and a features generator.')

        self.features = features

        if use_compound_names:
            self.compound_name = line[0]  # str
            line = line[1:]
        else:
            self.compound_name = None

        # 存储分子的SMILES字符串表示
        self.smiles = line[0]  # str
        # RDKit的Mol对象，用于分子结构操作
        self.mol = Chem.MolFromSmiles(self.smiles)

        # Generate additional features if given a generator
        # 如果说 features_generator 存在，例如：['morgan', 'rdkit_2d'] 那么其self.features 就是由多个特征向量拼接而成
        # 比如morgan得到的维度为[2048]  , rdkit_2d得到的维度为[200]. 那么self.features的维度为[2248]
        if self.features_generator is not None:
            # 分子特征向量（如摩根指纹、物化性质等）
            self.features = []

            for fg in self.features_generator:
                features_generator = get_features_generator(fg)
                if self.mol is not None and self.mol.GetNumHeavyAtoms() > 0:
                    self.features.extend(features_generator(self.mol))

            self.features = np.array(self.features)

        # 将self.features中的nan 变为0
        # Fix nans in features
        if self.features is not None:
            replace_token = 0
            self.features = np.where(np.isnan(self.features), replace_token, self.features)

        # Create targets
        # 预测目标值（如溶解度、毒性等）
        self.targets = [float(x) if x != '' else None for x in line[1:]]

    def set_features(self, features: np.ndarray):
        """
        设置分子的features （其实就是embedding）
        :param features：一个[ 1.0 ,2.1 ..] 一样的numpy数组
        """
        self.features = features

    def num_tasks(self) -> int:
        """
        Returns the number of prediction tasks.

        :return: The number of tasks.
        """
        return len(self.targets)

    def set_targets(self, targets: List[float]):
        """
        Sets the targets of a molecule.

        :param targets: A list of floats containing the targets.
        """
        self.targets = targets