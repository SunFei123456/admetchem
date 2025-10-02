#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn as nn

from utils.compound_tools import CompoundKit
from models_lib.basic_block import RBF


class AtomEmbedding(torch.nn.Module):
    """
    Atom Encoder
    """

    def __init__(self, atom_names, embed_dim, device):
        super(AtomEmbedding, self).__init__()
        self.atom_names = atom_names

        self.embed_list = nn.ModuleList()
        for name in self.atom_names:
            embed = nn.Embedding(CompoundKit.get_atom_feature_size(name) + 5, embed_dim).to(device)
            self.embed_list.append(embed)

    def forward(self, node_features):
        """
        Args:
            node_features(dict of tensor): node features.
        """
        out_embed = 0
        for i, name in enumerate(self.atom_names):
            out_embed += self.embed_list[i](node_features[i])
        return out_embed


class AtomFloatEmbedding(torch.nn.Module):
    """
    Atom Float Encoder
    """

    def __init__(self, atom_float_names, embed_dim, rbf_params=None, device=None):
        super(AtomFloatEmbedding, self).__init__()
        self.atom_float_names = atom_float_names

        if rbf_params is None:
            self.rbf_params = {
                'van_der_waals_radis': (torch.arange(1, 3, 0.2), 10.0),  # (centers, gamma)
                'partial_charge': (torch.arange(-1, 4, 0.25), 10.0),  # (centers, gamma)
                'mass': (torch.arange(0, 2, 0.1), 10.0),  # (centers, gamma)
            }
        else:
            self.rbf_params = rbf_params
            self.linear_list = nn.ModuleList()
            self.rbf_list = nn.ModuleList()
            for name in self.atom_float_names:
                centers, gamma = self.rbf_params[name]
                rbf = RBF(centers, gamma).to(device)
                self.rbf_list.append(rbf)
                linear = nn.Linear(len(centers), embed_dim).to(device)
                self.linear_list.append(linear)
                self.rbf_list.append(rbf)
                linear = nn.Linear(len(centers), embed_dim).to(device)
                self.linear_list.append(linear)

    def forward(self, feats):
        """
        Args:
            feats(dict of tensor): node float features.
        """
        out_embed = 0
        for i, name in enumerate(self.atom_float_names):
            x = feats[name]
            rbf_x = self.rbf_list[i](x)
            out_embed += self.linear_list[i](rbf_x)
        return out_embed


class BondEmbedding(nn.Module):
    """
    Bond Encoder
    """

    def __init__(self, bond_names, embed_dim, device):
        super(BondEmbedding, self).__init__()
        self.bond_names = bond_names

        self.embed_list = nn.ModuleList()
        for name in self.bond_names:
            embed = nn.Embedding(CompoundKit.get_bond_feature_size(name) + 5, embed_dim).to(device)
            self.embed_list.append(embed)
    #假设这里面的某个维度扩充。比如[1325,4] 变为[1325,256]
    def forward(self, edge_features):
        """
        Args:
            edge_features(dict of tensor): edge features.
        """
        out_embed = 0
        for i, name in enumerate(self.bond_names):
            out_embed += self.embed_list[i](edge_features[i].long())
        return out_embed

#处理化学键的连续型特征（如键长、键角等），通过径向基函数（RBF） 将连续值转换为高维表示，再通过线性层映射到嵌入空间。
class BondFloatRBF(nn.Module):
    """
    Bond Float Encoder using Radial Basis Functions
    """

    def __init__(self, bond_float_names, embed_dim, rbf_params=None, device=None):
        #embed_dim是嵌入空间大小，也就是输出维度
        super(BondFloatRBF, self).__init__()
        # 这里bonda_float_names为['bond_length']
        self.bond_float_names = bond_float_names

        if rbf_params is None:
            # torch.arange(0, 2, 0.1) 会生成一个从 0 开始、以 步长 0.1 递增、不包含终止值 2 的一维张量。
            # 有机分子中常见键长（如C-C、C=C、C≡C）范围集中在 0.7–1.8 Å 之间
            # 因此，0–2 Å 的区间覆盖了绝大多数化学键的合理范围，同时预留边界容错
            self.rbf_params = {
                'bond_length': (torch.arange(0, 2, 0.1).to(device), 10.0),  # (centers, gamma)
            }
        else:
            self.rbf_params = rbf_params

        self.linear_list = nn.ModuleList()
        self.rbf_list = nn.ModuleList()
        for name in self.bond_float_names:
            #1、RBF径向基函数的 中心点centers 以及 gamma ，创建RBF（非线性映射）
            centers, gamma = self.rbf_params[name]
            rbf = RBF(centers, gamma).to(device)
            self.rbf_list.append(rbf)
            #2、线性映射，最后输出
            linear = nn.Linear(len(centers), embed_dim).to(device)
            self.linear_list.append(linear)
    # bond_float_features 的维度是[1,1305] 表示 1305个原子的特征
    # 输出是[1,256] 嵌入表示
    def forward(self, bond_float_features):
        """
        Args:
            bond_float_features(dict of tensor): bond float features.
        """
        out_embed = 0
        # 一共有两步。第一步 通过RBF转换（非线性高维映射）  第二步 线性映射 nn.Linear(len(centers), embed_dim)
        for i, name in enumerate(self.bond_float_names):
            x = bond_float_features[i]
            rbf_x = self.rbf_list[i](x)
            out_embed += self.linear_list[i](rbf_x.float())
        return out_embed


class BondAngleFloatRBF(nn.Module):
    """
    Bond Angle Float Encoder using Radial Basis Functions
    """

    def __init__(self, bond_angle_float_names, embed_dim, rbf_params=None, device=None):
        super(BondAngleFloatRBF, self).__init__()
        self.bond_angle_float_names = bond_angle_float_names

        if rbf_params is None:
            self.rbf_params = {
                'bond_angle': (torch.arange(0, np.pi, 0.1).to(device), 10.0),  # (centers, gamma)
            }
        else:
            self.rbf_params = rbf_params

        self.linear_list = nn.ModuleList()
        self.rbf_list = nn.ModuleList()
        for name in self.bond_angle_float_names:
            centers, gamma = self.rbf_params[name]
            rbf = RBF(centers, gamma).to(device)
            self.rbf_list.append(rbf)
            linear = nn.Linear(len(centers), embed_dim).to(device)
            self.linear_list.append(linear)

    def forward(self, bond_angle_float_features):
        """
        Args:
            bond_angle_float_features(dict of tensor): bond angle float features.
        """
        out_embed = 0
        for i, name in enumerate(self.bond_angle_float_names):
            x = bond_angle_float_features[i]
            rbf_x = self.rbf_list[i](x)
            out_embed += self.linear_list[i](rbf_x.float())
        return out_embed