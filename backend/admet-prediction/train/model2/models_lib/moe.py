import torch
import torch.nn as nn
import torch.nn.functional as F
# 在 moe.py 中增强 MoEProjector
class MoEProjector(nn.Module):
    def __init__(self, input_dim, latent_dim, n_experts=4):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 2*latent_dim),
                nn.GELU(),
                nn.Linear(2*latent_dim, latent_dim)  # 更深的专家网络
            ) for _ in range(n_experts)
        ])
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, n_experts)  # 更复杂的门控网络
        )
        self.n_experts = n_experts

    def forward(self, x, custom_weights=None):
        if custom_weights is not None:
            gates = custom_weights
        else:
            gate_logits = self.gate(x)
            gates = F.softmax(gate_logits, dim=1)

        # 并行计算专家输出
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)

        # 添加专家负载均衡正则化
        if self.training and custom_weights is None:
            load = F.softmax(gate_logits.detach(), dim=1)
            expert_load = load.mean(dim=0)
            balance_loss = (expert_load.std() / expert_load.mean()) * 0.1  # 控制强度
        else:
            balance_loss = 0

        # 加权融合
        output = torch.einsum('be,bde->bd', gates, expert_outputs)

        return output, balance_loss