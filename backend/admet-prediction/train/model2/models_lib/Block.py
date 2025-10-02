import torch.nn as nn
import torch.nn.functional as F
import torch
from timm.models.layers import DropPath
class SparseMoE_Self_Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_experts=4,
                 num_heads=4,
                 top_k=2,
                 qkv_bias=False,
                 drop_ratio=0.,
                 update_rate=0.01
                 ):
        super(SparseMoE_Self_Attention, self).__init__()
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.top_k = top_k
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Define a separate qkv layer for each expert
        self.qkv_experts = nn.ModuleList([nn.Linear(dim, dim * 3, bias=qkv_bias) for _ in range(num_experts)])

        # Define gating network
        self.gating = nn.Linear(dim, num_experts)

        self.attn_drop = nn.Dropout(drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop_ratio)

        # ... [原有代码] ...
        self.update_rate = update_rate  # 偏置更新率
        self.expert_bias = nn.Parameter(torch.zeros(num_experts), requires_grad=False)  # 动态偏置项
    # 输入
    #　ｘ：（B,N） N是特征维度。
    # 输出
    # x:(B,N) 一模一样（经过稀疏 MoE 自注意力计算后的特征张量）
    def forward(self, x):
        B, N = x.shape

        #1、得到门控网络
        # Gating network to get the importance of each expert
        gate_scores = self.gating(x)  # Shape: (B, num_experts) 例如[64,4]
        gate_scores = F.softmax(gate_scores, dim=-1)  # Normalize scores to get probabilities 例如[64,4]
        # 添加动态偏置到门控分数
        gate_scores = gate_scores + self.expert_bias.unsqueeze(0)

        #2、TOP-K专家选择
        #输入　gate_scores: (B, num_experts)
        # 输出 top_k_values: (B, top_k)，每个样本的 top-k 权重值。维度[64,2] 比如 [[0.4659,0.2100],[0.3581,0.2320] ...]
        # top_k_indices: (B, top_k)，每个样本的 top-k 专家索引。比如[[0,1],[0,2]....]
        # Select top-k experts for each input
        top_k_values, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)

        # 统计专家负载（仅训练时更新）
        if self.training:
            expert_count = torch.zeros(self.num_experts, device=x.device)
            for i in range(self.top_k):
                expert_idx = top_k_indices[:, i]
                expert_count.scatter_add_(0, expert_idx, torch.ones_like(expert_idx, dtype=torch.float))

            # 动态调整偏置（负载越高，偏置越低）
            target_load = x.size(0) * self.top_k / self.num_experts
            load_error = target_load - expert_count
            self.expert_bias.data += self.update_rate * torch.sign(load_error)  # [4](@ref)
        #3、初始化 QKV q, k, v: 形状均为 (B, num_heads, head_dim) 比如[64,4,64]
        # Initialize q, k, v as zeros
        q = k = v = torch.zeros(B, self.num_heads, N // self.num_heads, device=x.device)

        #4、逐专家计算QKV
        # Only use the top-k experts for each input
        for i in range(self.top_k):
            #4.1 拿到每个专家的 编号 以及 每个专家的score
            expert_idx = top_k_indices[:, i]  # Shape: (B,) 比如[0,0,0,0,1....]
            gate_score = top_k_values[:, i].unsqueeze(-1).unsqueeze(-1) #维度64,1,1] 比如[[[0.4659]],[[0.3581]]...]

            #4.2 对每个样本，调用对应的专家网络计算 QKV  qkv的维度是 (B,1, 3*N)
            #  self.qkv_experts = nn.ModuleList([nn.Linear(dim, dim * 3, bias=qkv_bias) for _ in range(num_experts)])
            # Gather qkv for the selected experts
            qkv = torch.stack([self.qkv_experts[idx](x[b:b + 1])
                               for b, idx in enumerate(expert_idx)], dim=0)

            # # 重塑 QKV 为 (B, 3, num_heads, head_dim) ，然后通过permute 调整维度为 (3, B, num_heads, head_dim)
            qkv = qkv.reshape(B, 3, self.num_heads, N // self.num_heads).permute(1, 0, 2, 3)

            # 拆分 QKV
            q_i, k_i, v_i = qkv[0], qkv[1], qkv[2]

            # Accumulate q, k, v weighted by gate_score
            q += gate_score * q_i
            k += gate_score * k_i
            v += gate_score * v_i
        #5、自注意力计算
        # Self attention
        # q, k: 形状均为 (B, num_heads, head_dim)
        #attn: 注意力权重矩阵，形状为 (B, num_heads, head_dim, head_dim)。
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # attn = (attn - attn.amax(dim=-1, keepdim=True).detach()).softmax(dim=-1)
        assert not torch.isnan(attn).any(), "注意力输出含NaN"
        attn = self.attn_drop(attn)

        #6、输出投影
        x = (attn @ v).transpose(1, 2).reshape(B, N)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 drop_ratio=0.
                 ):
        super(Block, self).__init__()

        self.norm1 = norm_layer(dim)

        self.attn = SparseMoE_Self_Attention(dim, drop_ratio=drop_ratio)

        self.drop_path = DropPath(drop_ratio) if drop_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            drop=drop_ratio
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features,
                 act_layer=nn.GELU,
                 drop=0.):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SparseMoE_Cross_Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_experts=4,
                 num_heads=4,
                 top_k=2,
                 qkv_bias=False,
                 drop_ratio=0.,
                 update_rate=0.01
                 ):
        super(SparseMoE_Cross_Attention, self).__init__()
        self.num_heads = num_heads
        self.num_experts = num_experts
        self.top_k = top_k
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Define a separate qkv layer for each expert
        self.qkv_experts = nn.ModuleList([nn.Linear(dim, dim * 3, bias=qkv_bias) for _ in range(num_experts)])

        # Define gating network
        self.gating = nn.Linear(dim, num_experts)

        self.attn_drop = nn.Dropout(drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop_ratio)

        # ... [原有代码] ...
        self.update_rate = update_rate
        self.expert_bias = nn.Parameter(torch.zeros(num_experts), requires_grad=False)

    def forward(self, x, y):
        B, N = x.shape

        # Gating network to get the importance of each expert
        gate_scores = self.gating(x)  # Shape: (B, num_experts)
        gate_scores = F.softmax(gate_scores, dim=-1)  # Normalize scores to get probabilities
        gate_scores = gate_scores + self.expert_bias.unsqueeze(0)
        # Select top-k experts for each input
        top_k_values, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        # 统计专家负载（仅训练时更新）
        if self.training:
            expert_count = torch.zeros(self.num_experts, device=x.device)
            for i in range(self.top_k):
                expert_idx = top_k_indices[:, i]
                expert_count.scatter_add_(0, expert_idx, torch.ones_like(expert_idx, dtype=torch.float))

            # 动态调整偏置（负载越高，偏置越低）
            target_load = x.size(0) * self.top_k / self.num_experts
            load_error = target_load - expert_count
            self.expert_bias.data += self.update_rate * torch.sign(load_error)
            # Initialize q, k, v as zeros
        q = k = v = torch.zeros(B, self.num_heads, N // self.num_heads, device=x.device)

        # Only use the top-k experts for each input
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]  # Shape: (B,)
            gate_score = top_k_values[:, i].unsqueeze(-1).unsqueeze(-1)

            # Gather qkv for the selected experts
            qkv_x = torch.stack([self.qkv_experts[idx](x[b:b + 1])
                                 for b, idx in enumerate(expert_idx)], dim=0)
            qkv_y = torch.stack([self.qkv_experts[idx](y[b:b + 1])
                                 for b, idx in enumerate(expert_idx)], dim=0)

            qkv_x = qkv_x.reshape(B, 3, self.num_heads, N // self.num_heads).permute(1, 0, 2, 3)
            qkv_y = qkv_y.reshape(B, 3, self.num_heads, N // self.num_heads).permute(1, 0, 2, 3)

            q_i, k_i, v_i = qkv_y[0], qkv_x[1], qkv_x[2]

            # Accumulate q, k, v weighted by gate_score
            q += gate_score * q_i
            k += gate_score * k_i
            v += gate_score * v_i


        # Cross attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        # attn = (attn - attn.amax(dim=-1, keepdim=True).detach()).softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class Decoder(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 drop_ratio=0.
                 ):
        super(Decoder, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SparseMoE_Self_Attention(dim, drop_ratio=drop_ratio)

        self.cross_attn = SparseMoE_Cross_Attention(dim, drop_ratio=drop_ratio)

        self.drop_path = DropPath(drop_ratio) if drop_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            drop=drop_ratio
        )

    def forward(self, x, y):
        out = y + self.drop_path(self.cross_attn(x, self.norm1(y)))
        out = out + self.drop_path(self.mlp(self.norm2(y)))

        return out
