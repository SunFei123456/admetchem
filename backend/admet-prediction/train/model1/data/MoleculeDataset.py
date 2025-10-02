
# 管理完整的分子数据集，提供数据加载接口。
# 用于管理多个MoleculeDatapoint对象。
from random import random
from typing import List, Union, Callable

import numpy as np
from rdkit import Chem
from torch.utils.data.dataset import Dataset

from data.MoleculeDataPoint import MoleculeDatapoint
from data.scaler import StandardScaler


class MoleculeDataset(Dataset):
    """A MoleculeDataset contains a list of molecules and their associated features and targets."""

    def __init__(self, data: List[MoleculeDatapoint]):
        """
        Initializes a MoleculeDataset, which contains a list of MoleculeDatapoints (i.e. a list of molecules).

        :param data: A list of MoleculeDatapoints.
        """
        # self.data是一个列表，列表中都是MoleculeDatapoint对象
        self.data = data
        # 将第一个数据的args当作整个数据集的args
        self.args = self.data[0].args if len(self.data) > 0 else None
        # 特征标准化器
        self.scaler = None

    # 获取所有的化合物的名称。比如 ["Ethanol", "Caffeine"]
    def compound_names(self) -> List[str]:
        """
        Returns the compound names associated with the molecule (if they exist).

        :return: A list of compound names or None if the dataset does not contain compound names.
        """
        if len(self.data) == 0 or self.data[0].compound_name is None:
            return None

        return [d.compound_name for d in self.data]

    # 返回所有SMILES字符串列表，比如["CCO", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]
    def smiles(self) -> List[str]:
        """
        Returns the smiles strings associated with the molecules.

        :return: A list of smiles strings.
        """
        return [d.smiles for d in self.data]

    # 返回所有RDKit分子对象列表 [<rdkit.Chem.rdchem.Mol>, ...]
    def mols(self) -> List[Chem.Mol]:
        """
        Returns the RDKit molecules associated with the molecules.

        :return: A list of RDKit Mols.
        """
        return [d.mol for d in self.data]

    # 获取全部分子特征矩阵 [array([0.2, 1.5,...]),array([0.3, 1.2,...]), ...]
    def features(self) -> List[np.ndarray]:
        """
        Returns the features associated with each molecule (if they exist).

        :return: A list of 1D numpy arrays containing the features for each molecule or None if there are no features.
        """
        if len(self.data) == 0 or self.data[0].features is None:
            return None

        return [d.features for d in self.data]

    # 获取所有目标值矩阵 比如[[0.8, 1.2], [0.5, 0.9]]
    def targets(self) -> List[List[float]]:
        """
        Returns the targets associated with each molecule.

        :return: A list of lists of floats containing the targets.
        """
        return [d.targets for d in self.data]

    # 获取任务的数量 比如[[0.8, 1.2], [0.5, 0.9]] 那么任务数量就是2
    def num_tasks(self) -> int:
        """
        Returns the number of prediction tasks.

        :return: The number of tasks.
        """
        return self.data[0].num_tasks() if len(self.data) > 0 else None

    #获取self.features的特征维度
    def features_size(self) -> int:
        """
        Returns the size of the features array associated with each molecule.

        :return: The size of the features.
        """
        return len(self.data[0].features) if len(self.data) > 0 and self.data[0].features is not None else None
    # 随机打乱数据顺序，输入一个随机数种子seed
    def shuffle(self, seed: int = None):
        """
        Shuffles the dataset.

        :param seed: Optional random seed.
        """
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.data)
    # 特征标准化（使用Z-score标准化），使用一个标准化公式将该数据集中的所有self.features特征向量标准化归一化。：(features - mean) / std
    def normalize_features(self, scaler: StandardScaler = None, replace_nan_token: int = 0) -> StandardScaler:
        """
        Normalizes the features of the dataset using a StandardScaler (subtract mean, divide by standard deviation).

        If a scaler is provided, uses that scaler to perform the normalization. Otherwise fits a scaler to the
        features in the dataset and then performs the normalization.

        :param scaler: A fitted StandardScaler. Used if provided. Otherwise a StandardScaler is fit on
        this dataset and is then used.
        :param replace_nan_token: What to replace nans with.
        :return: A fitted StandardScaler. If a scaler is provided, this is the same scaler. Otherwise, this is
        a scaler fit on this dataset.
        """
        if len(self.data) == 0 or self.data[0].features is None:
            return None

        if scaler is not None:
            self.scaler = scaler

        elif self.scaler is None:
            features = np.vstack([d.features for d in self.data])
            self.scaler = StandardScaler(replace_nan_token=replace_nan_token)
            self.scaler.fit(features)

        for d in self.data:
            d.set_features(self.scaler.transform(d.features.reshape(1, -1))[0])

        return self.scaler

    #输入一个比如 [[0.7], [0.6]] 那么就会将原来的 目标值[[0.8, 1.2], [0.5, 0.9]] 给替换为[[0.7], [0.6]]
    def set_targets(self, targets: List[List[float]]):
        """
        Sets the targets for each molecule in the dataset. Assumes the targets are aligned with the datapoints.

        :param targets: A list of lists of floats containing targets for each molecule. This must be the
        same length as the underlying dataset.
        """
        assert len(self.data) == len(targets)
        for i in range(len(self.data)):
            self.data[i].set_targets(targets[i])

    def sort(self, key: Callable):
        """
        Sorts the dataset using the provided key.

        :param key: A function on a MoleculeDatapoint to determine the sorting order.
        """
        self.data.sort(key=key)

    #获取该数据集的数据总量，比如dataset = MoleculeDataset() ,len(dataset)
    def __len__(self) -> int:
        """
        Returns the length of the dataset (i.e. the number of molecules).

        :return: The length of the dataset.
        """
        return len(self.data)

    # 这里的item是一个index。是一个int类型，这也是该类能传入DataLoader迭代器的关键方法。
    def __getitem__(self, item) -> Union[MoleculeDatapoint, List[MoleculeDatapoint]]:
        """
        Gets one or more MoleculeDatapoints via an index or slice.

        :param item: An index (int) or a slice object.
        :return: A MoleculeDatapoint if an int is provided or a list of MoleculeDatapoints if a slice is provided.
        """
        return self.data[item]