import csv
import os
from argparse import Namespace
from typing import List
from logging import Logger

import dgl
import numpy as np
import pandas as pd
import torch
from dgl.dataloading import GraphDataLoader
from rdkit import Chem
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.MoleculeDataPoint import MoleculeDatapoint
from data.MoleculeDataset import MoleculeDataset
from data.data import MolGraphSet, Mol2HeteroGraph, Mol2AtomGraph, Mol2HeteroGraph2, Mol2AtomGraph2

# create_dataloader方法的返回值data_loader作用如下：
# data_loader = GraphDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
# 由于输出是data_loader
# for batched_graph, labels in data_loader:  # 每次迭代返回一个batch
#     # 模型前向传播、损失计算、反向传播等操作
#     outputs = model(batched_graph)
#     loss = criterion(outputs, labels)
#     返回的bactchedDGLGraph是一个超图，如下
#
# 举个例子，假设有两个图
# 图1：3个节点，节点特征维度为3；4条边。
# 图2：5个节点，节点特征维度为3；6条边。
#     BatchedDGLGraph(
#     num_nodes=8,
#     num_edges=10,
#     batch_size=2,  # 包含2个原始图
#     ndata={
#         'feat': tensor([[0.1, 0.2, 0.3],  # 图1的节点特征
#                         [0.4, 0.5, 0.6],
#                         [0.7, 0.8, 0.9],
#                         [1.0, 1.1, 1.2],  # 图2的节点特征
#                         [1.3, 1.4, 1.5],
#                         [1.6, 1.7, 1.8],
#                         [1.9, 2.0, 2.1],
#                         [2.2, 2.3, 2.4]]),
#         'batch': tensor([0,0,0,1,1,1,1,1])  # 节点所属图的索引
#     },
#     edata={
#         'edge_attr': tensor([...])  # 合并后的边特征
#     },
#     batch_num_nodes=[3,5],  # 各图的节点数
#     batch_num_edges=[4,6]   # 各图的边数
# )
# 再比如：有两个图
# 图1（g1）：
# 节点：
# 'a' 类型：3个原子，特征维度39（由 atom_features 生成）
# 'p' 类型：2个片段，特征维度128（需在 Mol2HeteroGraph 中补充）
# 边：
# ('a','b','a')：4条边（正反双向）
# ('p','r','p')：2条边（正反双向）
# ('a','j','p')：3条边（原子到片段）
# ('p','j','a')：3条边（片段到原子）
# 图2（g2）：
# 节点：
# 'a' 类型：5个原子，特征维度39
# 'p' 类型：3个片段，特征维度128
# 边：
# ('a','b','a')：6条边
# ('p','r','p')：4条边
# ('a','j','p')：5条边
# ('p','j','a')：5条边
# 节点 a 8个  p 5个 ('a','b','a') 10个  ('p','r','p') 6个 ('a','j','p') 8个 ('p','j','a') 8个
# BatchedDGLGraph(
#     num_nodes={'a': 8, 'p': 5},
#     num_edges={
#         ('a','b','a'): 10,
#         ('p','r','p'): 6,
#         ('a','j','p'): 8,
#         ('p','j','a'): 8
#     },
#     batch_size=2,
#     ndata={
#         'feat': {
#             'a': tensor([[0.1, ..., 0.39],  # 形状 (8, 39)
#                         [0.4, ..., 0.39],
#                         ...,  # 图1的3个原子 + 图2的5个原子
#                         [1.0, ..., 0.39]]),
#             'p': tensor([[1.0, ..., 1.128],  # 形状 (5, 128)
#                         ...,  # 图1的2个片段 + 图2的3个片段
#                         [2.0, ..., 1.128]])
#         }
#     },
#     batch_num_nodes={
#         'a': [3, 5],  # 图1的'a'节点数3，图2的'a'节点数5
#         'p': [2, 3]   # 图1的'p'节点数2，图2的'p'节点数3
#     },
#     batch_num_edges={
#         ('a','b','a'): [4, 6],  # 图1的边数4，图2的边数6
#         ('p','r','p'): [2, 4],
#         ...  # 其他边类型
#     }
# )
from features import load_features

def create_dataloader(args, filename, shuffle=True, train=True,collate_fn=None,val_size=None,test_size=None):

    data = get_data(path=os.path.join(args['path'], filename), args=args, logger=None)
    dataset = MoleculeDataset(data)

    if train:
        batch_size = args['batch_size']
    # elif val_size is not None:
    #     batch_size = min(val_size, len(dataset))
    # elif test_size is not None:
    #     batch_size = min(test_size, len(dataset))
    else:
        batch_size = min(4200, len(dataset))

    loader = DataLoader(dataset,
                              batch_size=batch_size,  # 典型值32-256
                              shuffle=True,  # 打乱顺序防止过拟合
                              collate_fn=collate_fn
                             )
    return loader

def create_dataloader_old(args, filename, shuffle=True, train=True):
    dataset = MolGraphSet(pd.read_csv(os.path.join(args['path'], filename)), args['target_names'])
    if train:
        batch_size = args['batch_size']
    else:
        batch_size = min(4200, len(dataset))

    dataloader = GraphDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader


# target_names是一个数组。比如["Class","Na"] 这个数组保存了 filename中的标签的名称
# filename是一个csv文件，里面保存了所有的化学分子smiles表达式
def create_dataloader_from_filename(filename, target_names=[], batch_size=64, shuffle=True, train=True):
    dataset = MolGraphSet(pd.read_csv(filename), target_names)
    if train:
        batch_size = batch_size
    else:
        batch_size = min(4200, len(dataset))
    dataloader = GraphDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader


#获取csv数据，将其转为MoleculeDataset
def get_data(path: str,
             skip_invalid_smiles: bool = True,
             args: Namespace = None,
             features_path: List[str] = None,
             max_data_size: int = None,
             use_compound_names: bool = None,
             logger: Logger = None) -> MoleculeDataset:
    """
    获取预训练文件将其变为一个data

    :param path:  CSV 文件的路径
    :param skip_invalid_smiles:  是否跳过并过滤掉无效的 SMILES 此处为True或者是False
    :param args: 参数
    :param features_path: 包含特征的文件路径列表。如果提供，则用于替代 args.features_path。比如 ["features/morgan.npy", "features/rdkit.npy"]
    :param max_data_size: 要加载的数据点的最大数量。比如 1000 (仅加载前1000行)
    :param use_compound_names: 文件是否包含除 SMILES 字符串之外的化合物名称。比如 True (CSV第一列为名称)
    :param logger: Logger
    :return: 一个包含SMILES字符串和目标值的MoleculeDataset，当需要时，还包含其他信息，如额外特征和化合物名称。
    """

    debug = logger.debug if logger is not None else print


    # Prefer explicit function arguments but default to args if not provided
    features_path = None


    max_data_size = max_data_size or float('inf')

    # Load features
    if features_path is not None:
        features_data = []
        for feat_path in features_path:
            # 加载分子指纹
            #   [
            #          array([[0.1, 0.2, ...], ...]),  # 摩根指纹 (1000,2048)
            #          array([[0.3, 0.5, ...], ...])   # RDKit 2D特征 (1000,200)
            #      ]
            features_data.append(load_features(feat_path))  # each is num_data x num_features
        features_data = np.concatenate(features_data, axis=1)
    else:
        features_data = None

    skip_smiles = set()

    # Load data
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        lines = []
        for line in reader:
            smiles = line[0]

            if smiles in skip_smiles:
                continue

            lines.append(line)

            if len(lines) >= max_data_size:
                break

        data = MoleculeDataset([
            MoleculeDatapoint(
                line=line,
                args=args,
                features=features_data[i] if features_data is not None else None,
                use_compound_names=use_compound_names
            ) for i, line in tqdm(enumerate(lines), total=len(lines))
        ])

    # Filter out invalid SMILES
    if skip_invalid_smiles:
        original_data_len = len(data)
        data = filter_invalid_smiles(data)

        if len(data) < original_data_len:
            debug(f'Warning: {original_data_len - len(data)} SMILES are invalid.')

    if data.data[0].features is not None:
        args.features_dim = len(data.data[0].features)

    return data

def filter_invalid_smiles(data: MoleculeDataset) -> MoleculeDataset:
    """
    Filters out invalid SMILES.

    :param data: A MoleculeDataset.
    :return: A MoleculeDataset with only valid molecules.
    """
    return MoleculeDataset([datapoint for datapoint in data
                            if datapoint.smiles != '' and datapoint.mol is not None
                            and datapoint.mol.GetNumHeavyAtoms() > 0])
def create_dgl_batch(smiles_batch: List[str]):
    graph,smiles = [],[]
    for smile in smiles_batch:
        mol = Chem.MolFromSmiles(smile)
        g = Mol2HeteroGraph(mol)
        graph.append(g)
        smiles.append(smile)
    return dgl.batch(graph),smiles

def create_atom_batch(smiles_batch: List[str]):
    graph, smiles = [], []
    for smile in smiles_batch:
        mol = Chem.MolFromSmiles(smile)
        g = Mol2AtomGraph(mol)
        graph.append(g)
        smiles.append(smile)
    return dgl.batch(graph), smiles
