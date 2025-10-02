#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
@author: Vencent_Wang
@contact: Vencent_Wang@outlook.com
@file: multi_modal.py
@time: 2023/8/13 20:05
@desc:
'''
import math
import torch

import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch import Tensor
from torch.nn import init
from torch.nn.parameter import Parameter
from torch_geometric.nn import global_mean_pool, GlobalAttention

from models_lib.Block import Block, Decoder
from models_lib.gnn_model import MPNEncoder
from models_lib.gem_model import GeoGNNModel
from models_lib.moe import MoEProjector
from models_lib.seq_model import TrfmSeq2seq

loss_type = {'class': nn.BCEWithLogitsLoss(reduction="none"), 'reg': nn.MSELoss(reduction="none")}


class Global_Attention(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.at = GlobalAttention(gate_nn=torch.nn.Linear(hidden_size, 1))

    def forward(self, x, batch):

        return self.at(x, batch)

class WeightFusion(nn.Module):

    def __init__(self, feat_views, feat_dim, bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(WeightFusion, self).__init__()
        self.feat_views = feat_views
        self.feat_dim = feat_dim
        self.weight = Parameter(torch.empty((1, 1, feat_views), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(int(feat_dim), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:

        return sum([input[i] * weight for i, weight in enumerate(self.weight[0][0])]) + self.bias


def quaternion_to_rotation_matrix(quaternion):
    q0 = quaternion[0]
    q1 = quaternion[1]
    q2 = quaternion[2]
    q3 = quaternion[3]
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
    return torch.stack([r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=-1).reshape(3, 3)
def get_random_rotation_3d(pos):
    random_quaternions = torch.randn(4).to(pos)
    random_quaternions = random_quaternions / random_quaternions.norm(dim=-1, keepdim=True)
    return torch.einsum("kj,ij->ki", pos, quaternion_to_rotation_matrix(random_quaternions))

class PreprocessBatch:
    def __init__(self, norm2origin=False, random_rotation=False) -> None:
        self.norm2origin = norm2origin
        self.random_rotation = random_rotation

    def process(self, atom_coords,node_id_all):
        if not self.norm2origin and not self.random_rotation:
            return
        pos = atom_coords
        if self.norm2origin:
            unique_ids, counts = torch.unique(node_id_all[0], return_counts=True)
            atom_counts_per_molecule = counts.tolist()
            pos_mean = global_mean_pool(pos,node_id_all[0])

            pos = pos - torch.repeat_interleave(pos_mean, counts, dim=0)
        if self.random_rotation:
            pos = get_random_rotation_3d(pos)
        atom_coords = pos
        return atom_coords

class Multi_modal(nn.Module):
    def __init__(self, args, compound_encoder_config, device):
        super().__init__()

        self.args = args
        self.device = device
        self.latent_dim = args.latent_dim
        self.batch_size = args.batch_size
        self.graph = args.graph
        self.sequence = args.sequence
        self.geometry = args.geometry
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.reduce_dim_512To256 = nn.Linear(512, 256)
        self.reduce_dim_768To256 = nn.Linear(768, 256)
        self.reduce_dim_512To2562 = nn.Linear(512, 256)
        if args.use_moe:
            #3D图模态
            self.norm_e1 = norm_layer(args.seq_hidden_dim)
            self.pos_drop_e1 = nn.Dropout(p=args.drop_ratio)
            self.pos_embed_e1 = nn.Parameter(torch.zeros(1, args.seq_hidden_dim)).to(device)
            dpr_e1 = [x.item() for x in torch.linspace(0, args.drop_ratio, args.depth_e1)]
            self.encoder_e1 = nn.Sequential(*[
                Block(dim=args.seq_hidden_dim,
                      mlp_ratio=4,
                      drop_ratio=dpr_e1[i],
                      )
                for i in range(args.depth_e1)
            ])
            #图模态
            self.norm_e2 = norm_layer(args.seq_hidden_dim)
            self.pos_embed_e2 = nn.Parameter(torch.zeros(1, args.seq_hidden_dim)).to(device)
            self.pos_drop_e2 = nn.Dropout(p=args.drop_ratio)
            dpr_e2 = [x.item() for x in torch.linspace(0, args.drop_ratio, args.depth_e1)]
            self.encoder_e2 = nn.Sequential(*[
                Block(dim=args.seq_hidden_dim,
                      mlp_ratio=4,
                      drop_ratio=dpr_e2[i],
                      )
                for i in range(args.depth_e1)
            ])
            #文本模态
            self.norm_e3 = norm_layer(args.seq_hidden_dim)
            self.pos_drop_e3 = nn.Dropout(p=args.drop_ratio)
            self.pos_embed_e3 = nn.Parameter(torch.zeros(1, args.seq_hidden_dim)).to(device)
            dpr_e3 = [x.item() for x in torch.linspace(0, args.drop_ratio, args.depth_e1)]
            self.encoder_e3 = nn.Sequential(*[
                Block(dim=args.seq_hidden_dim,
                      mlp_ratio=4,
                      drop_ratio=dpr_e3[i],
                      )
                for i in range(args.depth_e1)
            ])
        # CMPNN
        self.gnn = MPNEncoder(atom_fdim=args.gnn_atom_dim, bond_fdim=args.gnn_bond_dim,
                              hidden_size=args.gnn_hidden_dim, bias=args.bias, depth=args.gnn_num_layers,
                              dropout=args.dropout, activation=args.gnn_activation, device=device)
        # Transformer
        self.transformer = TrfmSeq2seq(input_dim=args.seq_input_dim, hidden_size=args.seq_hidden_dim,
                                       num_head=args.seq_num_heads, n_layers=args.seq_num_layers, dropout=args.dropout,
                                       vocab_num=args.vocab_num, device=device, recons=args.recons).to(self.device)
        # Geometric GNN
        self.compound_encoder = GeoGNNModel(args, compound_encoder_config, device)
        self.pro_num = args.pro_num
        # pro_num控制将不同模态投影到一个维度。当pro_num为3的时候就是将每一种模态用一个独立的Linear层进行投影
        # 当pro_num为1的时候，就是将每一种模态用一个Linear层进行投影
        if args.pro_num == 3:
            self.pro_seq = nn.Sequential(nn.Linear(args.seq_hidden_dim, self.latent_dim), nn.ReLU(inplace=True),
                                         nn.Linear(self.latent_dim, self.latent_dim)).to(device)
            self.pro_gnn = nn.Sequential(nn.Linear(args.gnn_hidden_dim, self.latent_dim), nn.ReLU(inplace=True),
                                         nn.Linear(self.latent_dim, self.latent_dim)).to(device)
            self.pro_geo = nn.Sequential(nn.Linear(args.geo_hidden_dim, self.latent_dim), nn.ReLU(inplace=True),
                                         nn.Linear(self.latent_dim, self.latent_dim)).to(device)
        elif args.pro_num == 1:
            self.pro_seq = nn.Sequential(nn.Linear(args.seq_hidden_dim, self.latent_dim), nn.ReLU(inplace=True),
                                         nn.Linear(self.latent_dim, self.latent_dim)).to(device)
            self.pro_gnn = self.pro_seq
            self.pro_geo = self.pro_seq

        # 添加跨模态门控网络
        if args.cross_modal_gate:
            self.cross_modal_gate = nn.Sequential(
                nn.Linear(
                    args.seq_hidden_dim + args.gnn_hidden_dim + args.geo_hidden_dim,
                    3 * args.n_experts  # 每个模态分配 n_experts 个权重
                ),
                nn.Softmax(dim=1)
            ).to(device)
        #损失
        self.entropy = loss_type[args.task_type]

        if args.pool_type == 'mean':
            self.pool = global_mean_pool
        else:
            self.pool = Global_Attention(args.seq_hidden_dim).to(self.device)

        # Fusion 融合模块

        # 当Fusion = 1，则直接拼接各模态特征。
        fusion_dim = args.gnn_hidden_dim * self.graph + args.seq_hidden_dim * self.sequence + \
                     args.geo_hidden_dim * self.geometry
        if self.args.fusion == 3:
        # 当Fusion = 3，则通过 WeightFusion学习各模态权重
            fusion_dim /= (self.graph + self.sequence + self.geometry)
            self.fusion = WeightFusion(self.graph + self.sequence + self.geometry, fusion_dim, device=self.device)
        elif self.args.fusion == 2 or self.args.fusion == 0:
        # 当Fusion = 2 或者 0 各模态特征逐元素相乘
            fusion_dim = args.seq_hidden_dim
        elif self.args.fusion == 4:
            self.decoder = Decoder(dim=args.seq_hidden_dim, mlp_ratio=4, drop_ratio=0., )
            fusion_dim = self.args.seq_hidden_dim * 2
        self.dropout = nn.Dropout(args.dropout)

        # MLP Classifier
        self.output_layer = nn.Sequential(nn.Linear(int(fusion_dim), int(fusion_dim)), nn.ReLU(), nn.Dropout(args.dropout),
                                          nn.Linear(int(fusion_dim), args.output_dim)).to(self.device)


        self.to(device)

    def forward_features_e1(self, x):

        pos_encoding = self.pos_embed_e1
        x = self.pos_drop_e1(x + pos_encoding)
        x = self.encoder_e1(x)
        x = self.norm_e1(x)
        return x
    def forward_features_e3(self, x):
        pos_encoding = self.pos_embed_e3
        x = self.pos_drop_e3(x + pos_encoding)
        x = self.encoder_e3(x)
        x = self.norm_e3(x)
        return x
    def forward_features_e2(self, x):


        # 1、self.pos_embed_e2 = nn.Parameter(torch.zeros(1, embed_dim)) 位置编码
        pos_encoding = self.pos_embed_e2

        # 2、pos_drop_e2 = nn.Dropout(p=drop_rate) 随机失活
        x = self.pos_drop_e2(x + pos_encoding)
        # 3、self.encoder_e2 = nn.Sequential(*[
        #     Block(dim=embed_dim,
        #           mlp_ratio=4,
        #           drop_ratio=dpr_e2[i],
        #           )
        #     for i in range(depth_e1)
        # ])
        x = self.encoder_e2(x)
        x = self.norm_e2(x)
        return x
    def label_loss(self, pred, label, mask):
        loss_mat = self.entropy(pred, label)
        return loss_mat.sum() / mask.sum()

    def cl_loss(self, x1, x2, T=0.1):
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss1).mean()
        return loss


    def loss_cal(self, x_list, pred, label, mask,moe_balance_loss, alpha=0.08,beta=0.05):
        #1、计算主任务的预测损失
        loss1 = self.label_loss(pred, label, mask)
        #2、计算对比损失
        loss2 = torch.tensor(0, dtype=torch.float).to(self.device)
        modal_num = len(x_list)
        #2.1 计算模态i与模态i-1的对比损失
        for i in range(modal_num):
            loss2 += self.cl_loss(x_list[i], x_list[i-1])
        #3、moe平衡损失
        loss3 = moe_balance_loss

        return loss1 + alpha * loss2 + beta * loss3, loss1, loss2,loss3

    def _forward(self, trans_batch_seq, seq_mask, batch_mask_seq, gnn_batch_graph, gnn_feature_batch, batch_mask_gnn,
                 graph_dict, node_id_all, edge_id_all, atom_coords):
        process = PreprocessBatch(True, False)
        atom_coords = process.process(atom_coords, node_id_all)
        x_list = list()
        cl_list = list()
        edge_index = graph_dict[0].edge_index
        moe_balance_loss = 0

        # 1、图模态处理
        if self.graph:
            node_gnn_x = self.gnn(gnn_batch_graph, gnn_feature_batch, batch_mask_gnn)  # 节点级特征提取
            graph_gnn_x = self.pool(node_gnn_x, batch_mask_gnn)  # 图级池化（均值/注意力）
            cl_list.append(self.pro_gnn(graph_gnn_x))  # 投影到对比学习空间
            if self.args.use_moe:
                graph_gnn_x = self.forward_features_e2(graph_gnn_x)
                if torch.isnan(graph_gnn_x).any():
                    print("Graph GNN output has NaN!")
            if self.args.norm:
                x_list.append(F.normalize(graph_gnn_x, p=2, dim=1))  # L2归一化
            else:
                x_list.append(graph_gnn_x)  # 原始特征

        # 2、序列模态处理
        if self.sequence:
            nloss, node_seq_x = self.transformer(trans_batch_seq)  # 序列编码（含重构损失）
            graph_seq_x = self.pool(node_seq_x[seq_mask], batch_mask_seq)  # 池化
            cl_list.append(self.pro_seq(graph_seq_x))
            if self.args.use_moe:
                graph_seq_x = self.forward_features_e3(graph_seq_x)
                if torch.isnan(graph_seq_x).any():
                    print("Sequence Transformer output has NaN!")
            if self.args.norm:
                x_list.append(F.normalize(graph_seq_x, p=2, dim=1))
            else:
                x_list.append(graph_seq_x)

        # 3、几何模态处理
        if self.geometry:
            node_repr, edge_repr = self.compound_encoder(graph_dict[0], graph_dict[1], node_id_all, edge_id_all,
                                                         atom_coords, edge_index)
            graph_geo_x = self.pool(node_repr, node_id_all[0])  # 几何图池化
            cl_list.append(self.pro_geo(graph_geo_x.to(self.device)))
            if self.args.use_moe:
                graph_geo_x = self.forward_features_e1(graph_geo_x)
                if torch.isnan(graph_geo_x).any():
                    print("Geometry Encoder output has NaN!")
            if self.args.norm:
                x_list.append(F.normalize(graph_geo_x, p=2, dim=1))
            else:
                x_list.append(graph_geo_x)

        # 4、多模态融合
        if self.args.fusion == 1:
            molecule_emb = torch.cat([temp for temp in x_list], dim=1)  # 拼接融合
        elif self.args.fusion == 2:
            molecule_emb = x_list[0].mul(x_list[1]).mul(x_list[2])  # 元素乘融合
        elif self.args.fusion == 3:
            molecule_emb = self.fusion(torch.stack(x_list, dim=0))  # 加权融合
        elif self.args.fusion == 4:
            image_graph_cross1 = self.decoder(graph_seq_x, graph_gnn_x)
            image_graph_cross2 = self.decoder(graph_gnn_x, graph_seq_x)
            cross_attention1 = torch.cat((image_graph_cross1, image_graph_cross2), dim=1)

            cross_attention1 = self.reduce_dim_512To256(cross_attention1)

            image_graph_fp_cross1 = self.decoder(cross_attention1, graph_geo_x)
            image_graph_fp_cross2 = self.decoder(graph_geo_x, cross_attention1)
            cross_attention2 = torch.cat((image_graph_fp_cross1, image_graph_fp_cross2), dim=1)

            molecule_emb = cross_attention2
        else:
            molecule_emb = torch.mean(torch.cat(x_list), dim=0, keepdim=True)  # 均值融合

        if not self.args.norm:
            molecule_emb = self.dropout(molecule_emb)
            # 改进的投影层处理

        # 5、返回预测 与 对比损失
        pred = self.output_layer(molecule_emb)
        return cl_list, pred, moe_balance_loss, molecule_emb  # 新增返回 molecule_emb

    # def forward_gnn(self, trans_batch_seq, seq_mask, batch_mask_seq, gnn_batch_graph, gnn_feature_batch, batch_mask_gnn,
    #                 graph_dict, node_id_all, edge_id_all, atom_coords):
    #     process = PreprocessBatch(True, False)
    #     atom_coords = process.process(atom_coords, node_id_all)
    #     x_list = []
    #     cl_list = []
    #     edge_index = graph_dict[0].edge_index
    #     moe_balance_loss = 0
    #
    #     # 仅处理图模态
    #     if self.graph:
    #         node_gnn_x = self.gnn(gnn_batch_graph, gnn_feature_batch, batch_mask_gnn)
    #         graph_gnn_x = self.pool(node_gnn_x, batch_mask_gnn)
    #         cl_list.append(self.pro_gnn(graph_gnn_x))
    #         if self.args.use_moe:
    #             graph_gnn_x = self.forward_features_e2(graph_gnn_x)
    #             if torch.isnan(graph_gnn_x).any():
    #                 print("Graph GNN output has NaN!")
    #         if self.args.norm:
    #             x_list.append(F.normalize(graph_gnn_x, p=2, dim=1))
    #         else:
    #             x_list.append(graph_gnn_x)
    #
    #     # 单模态直接使用GNN特征
    #     molecule_emb = x_list[0] if x_list else torch.tensor([]).to(self.device)
    #
    #     if not self.args.norm:
    #         molecule_emb = self.dropout(molecule_emb)
    #
    #     pred = self.output_layer(molecule_emb)
    #     return cl_list, pred, moe_balance_loss

    # def forward_geo(self, trans_batch_seq, seq_mask, batch_mask_seq, gnn_batch_graph, gnn_feature_batch, batch_mask_gnn,
    #                 graph_dict, node_id_all, edge_id_all, atom_coords):
    #     process = PreprocessBatch(True, False)
    #     atom_coords = process.process(atom_coords, node_id_all)
    #     x_list = []
    #     cl_list = []
    #     edge_index = graph_dict[0].edge_index
    #     moe_balance_loss = 0
    #
    #     # 仅处理几何模态
    #     if self.geometry:
    #         node_repr, edge_repr = self.compound_encoder(graph_dict[0], graph_dict[1], node_id_all, edge_id_all,
    #                                                      atom_coords, edge_index)
    #         graph_geo_x = self.pool(node_repr, node_id_all[0])
    #         cl_list.append(self.pro_geo(graph_geo_x.to(self.device)))
    #         if self.args.use_moe:
    #             graph_geo_x = self.forward_features_e1(graph_geo_x)
    #             if torch.isnan(graph_geo_x).any():
    #                 print("Geometry Encoder output has NaN!")
    #         if self.args.norm:
    #             x_list.append(F.normalize(graph_geo_x, p=2, dim=1))
    #         else:
    #             x_list.append(graph_geo_x)
    #
    #     # 单模态直接使用GEO特征
    #     molecule_emb = x_list[0] if x_list else torch.tensor([]).to(self.device)
    #
    #     if not self.args.norm:
    #         molecule_emb = self.dropout(molecule_emb)
    #
    #     pred = self.output_layer(molecule_emb)
    #     return cl_list, pred, moe_balance_loss

    # def forward_seq(self, trans_batch_seq, seq_mask, batch_mask_seq, gnn_batch_graph, gnn_feature_batch, batch_mask_gnn,
    #                 graph_dict, node_id_all, edge_id_all, atom_coords):
    #     process = PreprocessBatch(True, False)
    #     atom_coords = process.process(atom_coords, node_id_all)
    #     x_list = []
    #     cl_list = []
    #     edge_index = graph_dict[0].edge_index
    #     moe_balance_loss = 0
    #
    #     # 仅处理几何模态
    #     if self.sequence:
    #         nloss, node_seq_x = self.transformer(trans_batch_seq)  # 序列编码（含重构损失）
    #         graph_seq_x = self.pool(node_seq_x[seq_mask], batch_mask_seq)  # 池化
    #         cl_list.append(self.pro_seq(graph_seq_x))
    #         if self.args.use_moe:
    #             graph_seq_x = self.forward_features_e3(graph_seq_x)
    #             if torch.isnan(graph_seq_x).any():
    #                 print("Sequence Transformer output has NaN!")
    #         if self.args.norm:
    #             x_list.append(F.normalize(graph_seq_x, p=2, dim=1))
    #         else:
    #             x_list.append(graph_seq_x)
    #
    #     # 单模态直接使用GEO特征
    #     molecule_emb = x_list[0] if x_list else torch.tensor([]).to(self.device)
    #
    #     if not self.args.norm:
    #         molecule_emb = self.dropout(molecule_emb)
    #
    #     pred = self.output_layer(molecule_emb)
    #     return cl_list, pred, moe_balance_loss
    # 输入
    # trans_batch_Seq: 序列模态输入. 维度为 [batch_size, seq_len, seq_hidden_dim] batch_size就是一个batch有多少个分子。seq_len就是每个分子填充到了多少个原子，最后一个hidden_dim 就是一个one-hot 用来标注该原子是哪个原子。
    # seq_mask: 序列的mask，维度为[batch_size, seq_len] 用来标注一个分子中 哪些是用于填充的掩码。比如[[False,True,True.....]]
    # batch_mask_seq: 维度为[node_size] node_size是一个batch中的smiles对应的原子的个数。比如 H2O 的smiles写法为 O 。 用于表示 前几个原子属于哪一个分子（属于哪一个batch）比如[0,0,0,1,1,....,31]
    # gnn_batch_graph: 图数据结构
    # gnn_feature_batch: 图节点/边的特征。一般为None
    # batch_mask_gnn：图数据的批次掩码。比如H20有三个原子。其和batch_mask_seq同理。
    # graph_dict：几何模态的图结构（3D构象） 其是 一个tuple元组
    #       第一个元素 graph_dict[0] 是一个batch（原子图）。其中包含了x 原子特征矩阵 [num_atoms, atom_feature_dim]、edge_index 原子间的连接关系等
    #       第二个元素 graph_dict[1] 是一个batch（键图）。 其中包含了x 键特征矩阵，维度 [num_bonds, bond_feature_dim]、edge_index：键与键的连接关系，维度 [2, num_bond_edges]
    # node_id_all：几何图中节点的全局ID。
    # edge_id_all: 键图全局ID。
    # atom_coords: 原子图中的所有原子坐标。

    # def forward(self, trans_batch_seq, seq_mask, batch_mask_seq, gnn_batch_graph, gnn_feature_batch, batch_mask_gnn,
    #             graph_dict, node_id_all, edge_id_all, atom_coords):
    #     process = PreprocessBatch(True, False)
    #     atom_coords = process.process(atom_coords, node_id_all)
    #     x_list = list()
    #     cl_list = list()
    #     edge_index = graph_dict[0].edge_index
    #     moe_balance_loss = 0
    #
    #     # 1、图模态处理
    #     if self.graph:
    #         node_gnn_x = self.gnn(gnn_batch_graph, gnn_feature_batch, batch_mask_gnn)  # 节点级特征提取
    #         graph_gnn_x = self.pool(node_gnn_x, batch_mask_gnn)  # 图级池化（均值/注意力）
    #         cl_list.append(self.pro_gnn(graph_gnn_x))  # 投影到对比学习空间
    #         if self.args.use_moe:
    #             graph_gnn_x = self.forward_features_e2(graph_gnn_x)
    #             if torch.isnan(graph_gnn_x).any():
    #                 print("Graph GNN output has NaN!")
    #         if self.args.norm:
    #             x_list.append(F.normalize(graph_gnn_x, p=2, dim=1))  # L2归一化
    #         else:
    #             x_list.append(graph_gnn_x)  # 原始特征
    #
    #     # 2、序列模态处理
    #     if self.sequence:
    #         nloss, node_seq_x = self.transformer(trans_batch_seq)  # 序列编码（含重构损失）
    #         graph_seq_x = self.pool(node_seq_x[seq_mask], batch_mask_seq)  # 池化
    #         cl_list.append(self.pro_seq(graph_seq_x))
    #         if self.args.use_moe:
    #             graph_seq_x = self.forward_features_e3(graph_seq_x)
    #             if torch.isnan(graph_seq_x).any():
    #                 print("Sequence Transformer output has NaN!")
    #         if self.args.norm:
    #             x_list.append(F.normalize(graph_seq_x, p=2, dim=1))
    #         else:
    #             x_list.append(graph_seq_x)
    #
    #     # 3、几何模态处理
    #     if self.geometry:
    #         node_repr, edge_repr = self.compound_encoder(graph_dict[0], graph_dict[1], node_id_all, edge_id_all,
    #                                                      atom_coords, edge_index)
    #         graph_geo_x = self.pool(node_repr, node_id_all[0])  # 几何图池化
    #         cl_list.append(self.pro_geo(graph_geo_x.to(self.device)))
    #         if self.args.use_moe:
    #             graph_geo_x = self.forward_features_e1(graph_geo_x)
    #             if torch.isnan(graph_geo_x).any():
    #                 print("Geometry Encoder output has NaN!")
    #         if self.args.norm:
    #             x_list.append(F.normalize(graph_geo_x, p=2, dim=1))
    #         else:
    #             x_list.append(graph_geo_x)
    #
    #     # 4、多模态融合
    #     if self.args.fusion == 1:
    #         molecule_emb = torch.cat([temp for temp in x_list], dim=1)  # 拼接融合
    #     elif self.args.fusion == 2:
    #         molecule_emb = x_list[0].mul(x_list[1]).mul(x_list[2])  # 元素乘融合
    #     elif self.args.fusion == 3:
    #         molecule_emb = self.fusion(torch.stack(x_list, dim=0))  # 加权融合
    #     elif self.args.fusion == 4:
    #         image_graph_cross1 = self.decoder(graph_geo_x, graph_gnn_x)
    #         image_graph_cross2 = self.decoder(graph_gnn_x, graph_geo_x)
    #         cross_attention1 = torch.cat((image_graph_cross1, image_graph_cross2), dim=1)
    #
    #         cross_attention1 = self.reduce_dim_512To256(cross_attention1)
    #
    #         image_graph_fp_cross1 = self.decoder(cross_attention1, graph_seq_x)
    #         image_graph_fp_cross2 = self.decoder(graph_seq_x, cross_attention1)
    #         cross_attention2 = torch.cat((image_graph_fp_cross1, image_graph_fp_cross2), dim=1)
    #         molecule_emb = cross_attention2
    #         # image_graph_geo1 = self.decoder(graph_geo_x, graph_gnn_x)
    #         # image_graph_geo2 = self.decoder(graph_gnn_x, graph_geo_x)
    #         # cross_attention1 = torch.cat((image_graph_geo1, image_graph_geo2), dim=1)
    #         # cross_attention1 = self.reduce_dim_512To256(cross_attention1)
    #         #
    #         #
    #         # image_graph_seq1 = self.decoder(graph_seq_x, graph_gnn_x)
    #         # image_graph_seq2 = self.decoder(graph_gnn_x, graph_seq_x)
    #         # cross_attention2 = torch.cat((image_graph_seq1, image_graph_seq2), dim=1)
    #         # cross_attention2 = self.reduce_dim_512To256(cross_attention2)
    #         #
    #         # image_seq_geo1 = self.decoder(graph_geo_x, graph_seq_x)
    #         # image_seq_geo2 = self.decoder(graph_seq_x, graph_geo_x)
    #         # cross_attention3 = torch.cat((image_seq_geo1, image_seq_geo2), dim=1)
    #         # cross_attention3 = self.reduce_dim_512To256(cross_attention3)
    #         #
    #         # molecule_emb = torch.cat((cross_attention1, cross_attention2, cross_attention3), dim=1)
    #         # # molecule_emb = self.reduce_dim_768To256(molecule_emb)
    #
    #     else:
    #         molecule_emb = torch.mean(torch.cat(x_list), dim=0, keepdim=True)  # 均值融合
    #
    #     if not self.args.norm:
    #         molecule_emb = self.dropout(molecule_emb)
    #         # 改进的投影层处理
    #
    #     # 5、返回预测 与 对比损失
    #     pred = self.output_layer(molecule_emb)
    #     return cl_list, pred, moe_balance_loss