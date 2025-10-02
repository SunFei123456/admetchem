#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch.nn as nn

from torch_geometric.nn import GINEConv
from models_lib.gnn_block import GraphNorm, MeanPool, GraphPool
from models_lib.compound_encoder import AtomEmbedding, BondEmbedding, BondFloatRBF, BondAngleFloatRBF


class GeoGNNBlock(nn.Module):
    """
    GeoGNN Block
    """

    def __init__(self, embed_dim, dropout_rate, last_act, device):
        super(GeoGNNBlock, self).__init__()

        self.embed_dim = embed_dim
        self.last_act = last_act

        self.gnn = GINEConv(nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(),
                                          nn.Linear(embed_dim * 2, embed_dim))).to(device)
        self.norm = nn.LayerNorm(embed_dim).to(device)
        self.graph_norm = GraphNorm(device).to(device)
        if last_act:
            self.act = nn.ReLU().to(device)
        self.dropout = nn.Dropout(p=dropout_rate).to(device)
    # 输入
    # graph：图
    # node_hidden：节点隐藏层 维度为[num_size,hidden]
    # edge_hidden: 边隐藏层 维度为[edge_size,hidden]
    # node_id: 节点id [num_size] 说明每个节点是属于哪一个分子
    # edge_id: 边id [edge_size] 说明每个边是属于哪一个分子
    def forward(self, graph, node_hidden, edge_hidden, node_id, edge_id):
        """tbd"""
        #1、图卷积
        out = self.gnn(node_hidden, graph.edge_index, edge_hidden)
        #2、归一化
        out = self.norm(out) #层归一化
        out = self.graph_norm(graph, out, node_id, edge_id) #图归一化
        if self.last_act:
            out = self.act(out)
        #3、正则化
        out = self.dropout(out)
        #4、残差连接
        out = out + node_hidden
        return out

class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        output_sizes,
        use_layer_norm=False,
        activation=nn.ReLU,
        dropout=0.0,
        layernorm_before=False,
        use_bn=False,
        pos_drop=0.0
    ):
        super().__init__()
        self.drop = nn.Dropout(p=pos_drop)
        module_list = []
        if not use_bn:
            if layernorm_before:
                module_list.append(nn.LayerNorm(input_size).to("cuda:0"))

            if dropout > 0:
                module_list.append(nn.Dropout(dropout).to("cuda:0"))
            for size in output_sizes:
                module_list.append(nn.Linear(input_size, size).to("cuda:0"))
                if size != 1:
                    module_list.append(activation().to("cuda:0"))
                input_size = size
            if not layernorm_before and use_layer_norm:
                module_list.append(nn.LayerNorm(input_size).to("cuda:0"))
        else:
            for size in output_sizes:
                module_list.append(nn.Linear(input_size, size).to("cuda:0"))
                if size != 1:
                    module_list.append(nn.BatchNorm1d(size).to("cuda:0"))
                    module_list.append(activation().to("cuda:0"))
                input_size = size

        self.module_list = nn.ModuleList(module_list)
        self.reset_parameters()

    def reset_parameters(self):
        for item in self.module_list:
            if hasattr(item, "reset_parameters"):
                item.reset_parameters()

    def forward(self, x):
        for item in self.module_list:
            x = item(x)
            x = self.drop(x)
        return x

class GeoGNNModel(nn.Module):
    """
    The GeoGNN Model used in GEM.

    Args:
        model_config(dict): a dict of model configurations.
    """
    def __init__(self, args, model_config={}, device=None):
        super(GeoGNNModel, self).__init__()

        self.embed_dim = model_config.get('embed_dim', 32)
        self.dropout_rate = model_config.get('dropout_rate', 0.2)
        self.layer_num = model_config.get('layer_num', 8)
        self.readout = model_config.get('readout', 'mean')

        self.atom_names = model_config['atom_names']
        self.bond_names = model_config['bond_names']
        self.bond_float_names = model_config['bond_float_names']
        self.bond_angle_float_names = model_config['bond_angle_float_names']

        self.init_atom_embedding = AtomEmbedding(self.atom_names, self.embed_dim, device=device)
        self.init_bond_embedding = BondEmbedding(self.bond_names, self.embed_dim, device=device)
        self.init_bond_float_rbf = BondFloatRBF(self.bond_float_names, self.embed_dim, device=device)

        self.bond_embedding_list = nn.ModuleList()
        self.bond_float_rbf_list = nn.ModuleList()
        self.bond_angle_float_rbf_list = nn.ModuleList()
        self.atom_bond_block_list = nn.ModuleList()
        self.bond_angle_block_list = nn.ModuleList()
        self.pos_embedding = MLP(3, [self.embed_dim, self.embed_dim], pos_drop=0.0)
        self.dis_embedding = MLP(1, [self.embed_dim, self.embed_dim], pos_drop=0.0)
        for layer_id in range(self.layer_num):
            self.bond_embedding_list.append(
                BondEmbedding(self.bond_names, self.embed_dim, device=device))
            self.bond_float_rbf_list.append(
                BondFloatRBF(self.bond_float_names, self.embed_dim, device=device))
            self.bond_angle_float_rbf_list.append(
                BondAngleFloatRBF(self.bond_angle_float_names, self.embed_dim, device=device))
            self.atom_bond_block_list.append(
                GeoGNNBlock(self.embed_dim, self.dropout_rate, last_act=(layer_id != self.layer_num - 1), device=device))
            self.bond_angle_block_list.append(
                GeoGNNBlock(self.embed_dim, self.dropout_rate, last_act=(layer_id != self.layer_num - 1), device=device))

        # TODO: use self-implemented MeanPool due to pgl bug.
        if self.readout == 'mean':
            self.graph_pool = MeanPool()
        else:
            self.graph_pool = GraphPool(pool_type=self.readout)

        print('[GeoGNNModel] embed_dim:%s' % self.embed_dim)
        print('[GeoGNNModel] dropout_rate:%s' % self.dropout_rate)
        print('[GeoGNNModel] layer_num:%s' % self.layer_num)
        print('[GeoGNNModel] readout:%s' % self.readout)
        print('[GeoGNNModel] atom_names:%s' % str(self.atom_names))
        print('[GeoGNNModel] bond_names:%s' % str(self.bond_names))
        print('[GeoGNNModel] bond_float_names:%s' % str(self.bond_float_names))
        print('[GeoGNNModel] bond_angle_float_names:%s' % str(self.bond_angle_float_names))

    @property
    def node_dim(self):
        """the out dim of graph_repr"""
        return self.embed_dim

    @property
    def graph_dim(self):
        """the out dim of graph_repr"""
        return self.embed_dim

    def forward(self, atom_bond_graph, bond_angle_graph, node_id, edge_id,atom_coords,edge_index):
        """
        Build the network.
        """
        device = next(self.parameters()).device
        atom_coords = atom_coords.to(device)
        edge_index = edge_index.to(device)
        t = atom_bond_graph.x.T
        w = atom_bond_graph.x
        #1、初始化嵌入层
        node_hidden = self.init_atom_embedding(atom_bond_graph.x.T) # atom_bond_graph.x的维度为[393,7] atom_bond_graph.x.T的维度为[7,393] 而 node_hidden为[393,256] 本质上就是[原子数，hidden]
        # 对于离散特征，由于其对性质的影响是线性的，所以可以不用RBF。（类别差异是明确的（如单键≠双键），无需通过高斯函数转换。）
        bond_embed = self.init_bond_embedding(atom_bond_graph.edge_attr.T) # 键的离散特征（比如键类型） atom_bond_graph.edge_attr 为[987,4] 而 bond_num为[987,256]
        bond_e_embed = atom_bond_graph.edge_attr.T[len(self.bond_names):] # bond_e_embed维度为[1,987] 表示取到了所有边的最后一个特征。这个是键长，是一个float
        edge_hidden = bond_embed + self.init_bond_float_rbf(bond_e_embed) # 相加 本质上就是[键数，hidden]
        # 为什么要对键长或键角（连续型特征）进行RBF径向基神经网络呢
        # 第一、尺度敏感 键长单位（Å）与分子中其他特征（如键角90°）数值差异巨大
        # 第二、非线性：键长从1.54Å（单键）缩短至1.34Å时，键能从347 kJ/mol跃升至612 kJ/mol，非比例变化（对于分子性质预测的非线性变化）
        # 第三、局部敏感：当键长处于1.3–1.4Å区间时，0.1Å变化可能改变键级（也就是可能让 化学键类型发生改变： 单键→双键）
        # 所以 我们利用径向基函数 来只关注 键长的变化。
        # 第一、解决了局部敏感性：每个高斯函数仅对中心点附近的值敏感，可捕捉关键阈值效应（如氢键的临界长度）
        # 第二、归一化输出：RBF输出值域为[0,1]，避免尺度差异问题。


        node_hidden_list = [node_hidden] #该列表长度为1 元素是一个tensor
        edge_hidden_list = [edge_hidden] #该列表长度为1 元素是一个tensor

        #2、多层级处理
        for layer_id in range(self.layer_num):
            atom_coords = atom_coords.to(node_hidden_list[layer_id].device)
            edge_index = edge_index.to(edge_hidden_list[layer_id].device)
            node_hidden_list[layer_id] = node_hidden_list[layer_id] + self.pos_embedding(atom_coords)
            row = edge_index[0]  # 维度是一个2724的列表
            col = edge_index[1]  # 维度是一个2724的列表
            sent_pos = atom_coords[row]  # 维度[2724,3]
            received_pos = atom_coords[col]  # 维度[2724,3]
            length = (sent_pos - received_pos).norm(dim=-1).unsqueeze(-1)  # 维度[2724,1]
            edge_hidden_list[layer_id] = edge_hidden_list[layer_id] + self.dis_embedding(length)

            # 2.1 原子-键图处理 ，直接通过第i层的图卷积，然后输出到node_hidden
            node_hidden = self.atom_bond_block_list[layer_id](
                atom_bond_graph,
                node_hidden_list[layer_id],
                edge_hidden_list[layer_id], node_id[0], edge_id[0])

            # 2.2 键角图处理
            cur_edge_hidden = self.bond_embedding_list[layer_id](atom_bond_graph.edge_attr.T) #将atom_bond_graph.edge_attr 的[987,4] 变为 [987,256]
            cur_edge_hidden = cur_edge_hidden + self.bond_float_rbf_list[layer_id](atom_bond_graph.edge_attr.T[len(self.bond_names):])
            cur_angle_hidden = self.bond_angle_float_rbf_list[layer_id](bond_angle_graph.edge_attr.T)
            edge_hidden = self.bond_angle_block_list[layer_id](
                bond_angle_graph,
                cur_edge_hidden,
                cur_angle_hidden, node_id[1], edge_id[1])

            # 2.3 保存当前层输出
            node_hidden_list.append(node_hidden)
            edge_hidden_list.append(edge_hidden)

        #3、生成
        node_repr = node_hidden_list[-1] #最终原子表示
        edge_repr = edge_hidden_list[-1] #最终键表示
        # graph_repr = self.graph_pool(atom_bond_graph, node_repr, node_id[0], edge_id[0])
        return node_repr, edge_repr

