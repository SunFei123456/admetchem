"""
    PModel 用来预测最后结果

"""
import torch
from torch import nn


class PModel(nn.Module):
    def __init__(self, model1, model2, hidden_dim=256, output_dim=1):
        super().__init__()
        self.model1 = model1
        self.model2 = model2

        # 冻结预训练模型参数
        for param in self.model1.parameters():
            param.requires_grad = False
        for param in self.model2.parameters():
            param.requires_grad = False

        # 假设两个模型的输出维度都是128
        self.fc = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, bg):
        emb1 = self.model1(bg)  # [batch_size, 128]
        emb2 = self.model2(bg)  # [batch_size, 128]
        fused = torch.cat([emb1, emb2], dim=1)  # [batch_size, 256]
        return self.fc(fused)