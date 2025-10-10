
from argparse import Namespace

import dgl
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from dgl import function as fn
# 兼容不同版本的DGL
if not hasattr(fn, 'copy_src'):
    fn.copy_src = fn.copy_u
from functools import partial
import copy

import math

from data.util import create_dgl_batch, create_atom_batch

from typing import Union, List
from torch import Tensor
from torch import nn, sum
from model.GeaNet import GEANet
from torch.nn import init


# dgl graph utils
# 输入tensor = [a, b, c, d]
# 输出 [b, a, d, c]
# 用于获得反向边
def reverse_edge(tensor):
    n = tensor.size(0)
    assert n % 2 == 0
    delta = torch.ones(n).type(torch.long)
    delta[torch.arange(1, n, 2)] = -1
    return tensor[delta + torch.tensor(range(n))]


def del_reverse_message(edge, field):
    """for g.apply_edges"""
    # edge.src表示该边的源节点的field属性。

    return {'m': edge.src[field] - edge.data['rev_h']}


# 通过注意力机制进行聚合
# 调用attn这个注意力
# 返回一个field
# {
#     "a" : [[0.1,2]]
# }
def add_attn(node, field, attn):
    feat = node.data[field].unsqueeze(1)
    return {field: (attn(feat, node.mailbox['m'], node.mailbox['m'] + feat)).squeeze(1)}


# nn modules

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# 注意力机制
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    # p_attn = F.softmax(scores, dim = -1).masked_fill(mask, 0)  # 不影响
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


# 多头注意力
class MultiHeadedAttention(nn.Module):
    # d_model: 输入特征维度
    def __init__(self, h, d_model, dropout=0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        # 将输入分成 h 个头，每个头的维度为 d_k = d_model // h
        self.d_k = d_model // h
        # 头的数量
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class Node_GRU(nn.Module):
    """GRU for graph readout. Implemented with dgl graph"""

    def __init__(self, hid_dim, bidirectional=True,view="apj"):
        super(Node_GRU, self).__init__()
        self.hid_dim = hid_dim
        if bidirectional:  # 使用双向GRU
            self.direction = 2
        else:
            self.direction = 1
        # 多头注意力机制
        self.att_mix = MultiHeadedAttention(6, hid_dim)
        self.gru = nn.GRU(hid_dim, hid_dim, batch_first=True,
                          bidirectional=bidirectional)
        self.view = view
    # 将批量图中的节点特征拆分并填充到相同的尺寸。
    # 这是什么意思呢
    # 就是获得一个列表
    # [
    #     [
    #         [0,1,1,2],
    #         [2,1,1,2]
    #         [4,1,1,1]
    #     ],
    #     [
    #         [4,3,2,2],
    #         [4,1,2,3]
    #         [0,0,0,0]
    #     ]
    #
    # ]
    # 这是什么意思呢，就是表示在batch中，每张图的结点数量不同，首先，如果节点数量不同，就填充0向量，然后将每个节点的特征提取出来，变成如上图一样的list
    def split_batch(self, bg, ntype, field, device):
        # 获得该节点特征，是一个二维数组。
        hidden = bg.nodes[ntype].data[field]
        # node_size为结点数量,是一个一维数组，代表每一个图的节点数量，可以和上一个变量hidden配合获得每个图节点的特征
        node_size = bg.batch_num_nodes(ntype)
        # 获得起始索引，根据node_size 获得每个图在hidden中的起始位置
        start_index = torch.cat([torch.tensor([0], device=device), torch.cumsum(node_size, 0)[:-1]])
        # 找到该类型节点的最大数量
        max_num_node = max(node_size)
        # padding
        hidden_lst = []
        for i in range(bg.batch_size):
            start, size = start_index[i], node_size[i]
            assert size != 0, size
            # 截断
            cur_hidden = hidden.narrow(0, start, size)
            # 填充0
            cur_hidden = torch.nn.ZeroPad2d((0, 0, 0, max_num_node - cur_hidden.shape[0]))(cur_hidden)
            # 增加了一个维度
            hidden_lst.append(cur_hidden.unsqueeze(0))

        hidden_lst = torch.cat(hidden_lst, 0)

        return hidden_lst




    # 最大池化
    def max_pooling(self,a_pharmj, p_pharmj):
        return torch.max(a_pharmj, dim=1).values + torch.max(p_pharmj, dim=1).values



    class LinearAggregation(nn.Module):
        def __init__(self, a_input_dim,p_input_dim, output_dim):
            super().__init__()
            self.mlp_a = nn.Sequential(
                nn.Linear(a_input_dim, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim)
            )
            self.mlp_p = nn.Sequential(
                nn.Linear(p_input_dim, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim)
            )

        def forward(self, a_pharmj, p_pharmj):
            # 获取输入张量的设备
            device = a_pharmj.device
            # 将线性层迁移到输入张量所在的设备上
            self.mlp_a.to(device)
            self.mlp_p.to(device)
            a_transformed = self.mlp_a(a_pharmj)
            p_transformed = self.mlp_p(p_pharmj)
            return a_transformed.sum(dim=1)+ p_transformed.sum(dim=1)

    def forward(self, bg, suffix='h'):
        """
        bg: dgl.Graph (batch)
        hidden states of nodes are supposed to be in field 'h'.
        """
        self.suffix = suffix
        device = bg.device


        a_pharmj = self.split_batch(bg, 'a', f'f_{suffix}', device)
        h = a_pharmj

        # 将每个图中的每个原子的特征的最大值组成一个向量最大的向量特征取出
        hidden = h.max(1)[0].unsqueeze(0).repeat(self.direction, 1, 1)
        h, hidden = self.gru(h, hidden)

        # unpadding and reduce (mean) h: batch * L * hid_dim
        graph_embed = []

        node_size = bg.batch_num_nodes('a')
        start_index = torch.cat([torch.tensor([0], device=device), torch.cumsum(node_size, 0)[:-1]])
        for i in range(bg.batch_size):
            start, size = start_index[i], node_size[i]
            # 取每一个图的结点的平均值作为图嵌入
            graph_embed.append(h[i, :size].view(-1, self.direction * self.hid_dim).mean(0).unsqueeze(0))
        graph_embed = torch.cat(graph_embed, 0)

        return graph_embed


class MVMP(nn.Module):
    def __init__(self, msg_func=add_attn, hid_dim=300, depth=3, view='aba', suffix='h', act=nn.ReLU(), GEANet_cfg=None):
        """
        MultiViewMassagePassing
        view: a, ap, apj
        suffix: filed to save the nodes' hidden state in dgl.graph.
                e.g. bg.nodes[ntype].data['f'+'_junc'(in ajp view)+suffix]
        """
        super(MVMP, self).__init__()
        # view 有a ap apj三种，就是三种视图
        self.view = view
        self.depth = depth
        self.suffix = suffix
        # 消息传递函数，主要是用来调用attention的
        self.msg_func = msg_func
        # 激活函数
        self.act = act
        # 同构边
        self.homo_etypes = [('a', 'b', 'a')]
        # 异构边
        self.hetero_etypes = []
        # 节点类型
        self.node_types = ['a']


        # 一个字典
        self.attn = nn.ModuleDict()
        # 对于每一个边创建一个注意力
        # ｛
        #   ‘aba’ : MultiHeadedAttention(4,hid_dim)
        #
        # ｝
        for etype in self.homo_etypes + self.hetero_etypes:
            self.attn[''.join(etype)] = MultiHeadedAttention(4, hid_dim)

        # 一个字典，用于存储每个同构边类型下的消息传递模块列表。
        # {
        #   'aba': [nnLinear,nnLinear,........, 一共右deepth个]
        #
        # }
        self.mp_list = nn.ModuleDict()
        for edge_type in self.homo_etypes:
            self.mp_list[''.join(edge_type)] = nn.ModuleList([nn.Linear(hid_dim, hid_dim) for i in range(depth - 1)])

        # self.node_last_layer：一个字典，用于存储每个节点类型的最后一层线性变换。
        # ｛
        #   'a' : nn.Linear
        #
        #
        # ｝
        self.node_last_layer = nn.ModuleDict()
        for ntype in self.node_types:
            self.node_last_layer[ntype] = nn.Linear(3 * hid_dim, hid_dim)

        # 定义输入维度
        dim = 300

        # 实例化 GEANet 对象
        self.geanet = GEANet(dim, GEANet_cfg)

    def update_edge(self, edge, layer):
        return {'h': self.act(edge.data['x'] + layer(edge.data['m']))}

    def update_node(self, node, field, layer):
        return {field: layer(torch.cat([node.mailbox['mail'].sum(dim=1),
                                        node.data[field],
                                        node.data['f']], 1))}

    def init_node(self, node):
        return {f'f_{self.suffix}': node.data['f'].clone()}

    def init_edge(self, edge):
        return {'h': edge.data['x'].clone()}

    def forward(self, bg):
        suffix = self.suffix
        # 初始化前
        # node_a = {'f': [1.0, 2.0]}
        # node_b = {'f': [3.0, 4.0]}
        # node_p = {'f': [5.0, 6.0]}
        # #初始化后
        #
        # node_a = {'f': [1.0, 2.0], 'f_h': [1.0, 2.0]}
        # node_b = {'f': [3.0, 4.0], 'f_h': [3.0, 4.0]}
        # node_p = {'f': [5.0, 6.0], 'f_h': [5.0, 6.0]}
        for ntype in self.node_types:
            if ntype != 'junc':
                bg.apply_nodes(self.init_node, ntype=ntype)

        # 初始化之后
        # edge_e1 = {'x': [0.1, 0.2], 'h': [0.1, 0.2]}
        # edge_e2 = {'x': [0.3, 0.4], 'h': [0.3, 0.4]}
        # bg.edge[('a','b','a'))].data['h']
        for etype in self.homo_etypes:
            bg.apply_edges(self.init_edge, etype=etype)



        update_funcs = {
            e: (fn.copy_e('h', 'm'), partial(self.msg_func, attn=self.attn[''.join(e)], field=f'f_{suffix}')) for e in
            self.homo_etypes}
        update_funcs.update({e: (fn.copy_src(f'f_junc_{suffix}', 'm'),
                                 partial(self.msg_func, attn=self.attn[''.join(e)], field=f'f_junc_{suffix}')) for e in
                             self.hetero_etypes})
        # message passing

        for i in range(self.depth - 1):
            bg.multi_update_all(update_funcs, cross_reducer='sum')
            for edge_type in self.homo_etypes:
                bg.edges[edge_type].data['rev_h'] = reverse_edge(bg.edges[edge_type].data['h'])
                bg.apply_edges(partial(del_reverse_message, field=f'f_{suffix}'), etype=edge_type)
                bg.apply_edges(partial(self.update_edge, layer=self.mp_list[''.join(edge_type)][i]), etype=edge_type)

        for edge_type in self.homo_etypes:
            if 'a' in edge_type:
                # 获取节点和边的特征
                node_a_features = bg.nodes['a'].data[f'f_{suffix}']
                edge_aba_features = bg.edges[('a', 'b', 'a')].data['h']
                # 将节点和边特征传入 geanet 的 forward 方法
                updated_node_a, updated_edge_aba = self.geanet(node_a_features, edge_aba_features)
                # 更新图中的节点和边特征
                bg.nodes['a'].data[f'f_{suffix}'] = updated_node_a
                bg.edges[('a', 'b', 'a')].data['h'] = updated_edge_aba
            else:
                node_p_features = bg.nodes['p'].data[f'f_{suffix}']
                edge_prp_features = bg.edges[('p', 'r', 'p')].data['h']
                updated_node_p, updated_edge_prp = self.geanet(node_p_features, edge_prp_features)
                bg.nodes['p'].data[f'f_{suffix}'] = updated_node_p
                bg.edges[('p', 'r', 'p')].data['h'] = updated_edge_prp

        # last update of node feature
        update_funcs = {e: (
        fn.copy_e('h', 'mail'), partial(self.update_node, field=f'f_{suffix}', layer=self.node_last_layer[e[0]])) for e
                        in self.homo_etypes}
        bg.multi_update_all(update_funcs, cross_reducer='sum')

        # last update of junc feature
        bg.multi_update_all({e: (fn.copy_src(f'f_junc_{suffix}', 'mail'),
                                 partial(self.update_node, field=f'f_junc_{suffix}',
                                         layer=self.node_last_layer['junc'])) for e in self.hetero_etypes},
                            cross_reducer='sum')


class AModelEncoder(nn.Module):
    def __init__(self, args):
        super(AModelEncoder, self).__init__()
        from model.util import get_func
        hid_dim = args['hid_dim']
        self.act = get_func(args['act'])
        self.depth = args['depth']
        self.view = args['view']
        self.hid_dim = hid_dim
        # init
        # atom view
        self.w_atom = nn.Linear(args['atom_dim'], hid_dim)
        self.w_bond = nn.Linear(args['bond_dim'], hid_dim)


        # 定义一个配置字典
        GEANet_cfg = {
            "n_heads": args['Gea_n_heads'],
            "shared_unit": True,
            "edge_unit": True,
            "unit_size": args['Gea_unit_size']
        }

        # 将字典转换为对象属性访问
        class Config:
            def __init__(self, cfg):
                self.__dict__.update(cfg)

        # 创建配置对象
        GEANet_cfg = Config(GEANet_cfg)
        ## define the view during massage passing
        self.mp_aug = MVMP(msg_func=add_attn, hid_dim=hid_dim, depth=self.depth, view=self.view, suffix='aug', act=self.act,
                           GEANet_cfg=GEANet_cfg)

        ## readout
        self.readout_attn = Node_GRU(hid_dim,view=self.view)

        ## predict
        self.W_o = nn.Sequential(nn.Linear(2 * hid_dim, hid_dim),
                                 self.act,
                                 nn.Linear(hid_dim, hid_dim),

                                 )
        self.dropout = nn.Dropout(args["dropout"])
        self.initialize_weights()

    def initialize_weights(self):
        for param in self.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)

    def init_feature(self, bg):
        bg.nodes['a'].data['f'] = self.act(self.w_atom(bg.nodes['a'].data['f']))
        bg.edges[('a', 'b', 'a')].data['x'] = self.act(self.w_bond(bg.edges[('a', 'b', 'a')].data['x']))

    def get_atom_embeddings(self, bg) :
        """
        获取批次中每个分子的原子节点嵌入
        Args:
            bg: dgl图批次
        Returns:
            三维张量，形状为 [batch_size, max_atoms, hid_dim]
        """
        # 确保执行了特征初始化和消息传递
        self.init_feature(bg)
        self.mp_aug(bg)

        # 获取所有原子嵌入 [总原子数, hid_dim]
        atom_embeddings = bg.nodes['a'].data['f_aug']

        # 获取每个分子的原子数量
        atom_counts = bg.batch_num_nodes('a')
        max_atoms = atom_counts.max().item()
        batch_size = bg.batch_size

        # 创建填充后的三维张量
        padded_embeddings = torch.zeros(
            (batch_size, max_atoms, self.hid_dim),
            device=atom_embeddings.device,
            dtype=atom_embeddings.dtype
        )

        # 填充张量
        start_idx = 0
        for i in range(batch_size):
            num_atoms = atom_counts[i].item()
            padded_embeddings[i, :num_atoms] = atom_embeddings[start_idx:start_idx + num_atoms]
            start_idx += num_atoms

        return padded_embeddings

    def forward(self, bg):
        """
        Args:
            bg: a batch of graphs
        """


        self.init_feature(bg)
        #如果view='apj' 那么就是需要三视图合并的
        self.mp_aug(bg)
        #如果view='apj' 那么就是需要三视图合并的
        embed_aug = self.readout_attn(bg, 'aug')

        embed_aug = self.W_o(embed_aug)

        embed_aug = self.dropout(embed_aug)
        return embed_aug


# 这里多一层 主要是对原始的smiles进行处理，让其变为一个AModelEncoder的输入。
class AModel(nn.Module):
    def __init__(self,
                 args: Namespace,
                 graph_input: bool = False):
        super(AModel, self).__init__()


        self.graph_input = graph_input
        self.args = args

        # 这样的话可以选择，仅加载编码器部分的参数，或者 加载全部参数。
        self.encoder = AModelEncoder(self.args)

    def get_atom_embeddings(self, batch) -> torch.Tensor:
        """
        获取批次中每个分子的原子节点嵌入
        Args:
            batch: 输入数据（SMILES字符串列表或预处理好的图）
        Returns:
            三维张量，形状为 [batch_size, max_atoms, hid_dim]
        """
        if not self.graph_input:
            bg, smiles = create_atom_batch(batch)
            bg = bg.to(self.args["device"])
        else:
            bg = batch

        return self.encoder.get_atom_embeddings(bg)

    def forward(self,  batch
                ) -> torch.FloatTensor:
        if not self.graph_input:  # if features only, batch won't even be used
            batch,smiles = create_atom_batch(batch)
            batch = batch.to(self.args["device"])
        output = self.encoder.forward(batch)
        return output

