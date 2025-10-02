import torch
from torch import nn
import torch.nn.functional as F

class NCESoftmaxLoss(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, similarity):
        # 针对相似度矩阵  计算原始batch大小
        batch_size = similarity.size(0) // 2

        # 生成标签：对于每个样本i，正样本是i+batch_size的位置
        label = torch.tensor([(batch_size + i) % (batch_size * 2) for i in range(batch_size * 2)]).cuda().long()
        # 计算交叉熵损失

        # 假设
        # similarity = [
        #     [-1e12, 19.492, 19.998, 19.490],
        #     [19.492, -1e12, 19.490, 19.998],
        #     [19.998, 19.490, -1e12, 19.490],
        #     [19.490, 19.998, 19.490, -1e12]
        # ]
        #
        # label = [2, 3, 0, 1]

        # 那么对于 第一个向量 [-1e12, 19.492, 19.998, 19.490] ，其中正样本就是 2号索引 ，也就是19.998是正样本，也就是zi_1和 zj_3的相似度作为正样本其余是负样本。并且去除掉自身的影响。

        loss = self.criterion(similarity, label)
        return loss
class NodeContrastiveLoss(nn.Module):
    def __init__(self, loss_computer: str, temperature: float) -> None:
        super().__init__()

        if loss_computer == 'nce_softmax':
            self.loss_computer = NCESoftmaxLoss()
        else:
            raise NotImplementedError(f"Loss Computer {loss_computer} not Support!")
        self.temperature = temperature

    #atom_embed是一个三维的向量
    #第一个维度是batch的大小，第二个维度是每个样本（化学分子）的原子数量（该值是所有样本的原子数量的最大值），第三个维度是每个分子中原子的embedding
    #当一个batch中向量全为0的时候代表 已经没有原子了
    #fragment_embed是一个三维向量
    #第一个维度是batch的大小，第二个维度是每个样本（化学分子）中 由不同的部分原子组合成的片段数量（同理，是最大值），第三个维度是fragment片段的embeddings
    #当一个batch中向量全为0就代表已经没有片段了
    #最后一个index就是一个列表，其中每个元素是一个字典，字典的key是片段的索引，字典的value是一个列表，该列表里面包含了原子的索引

    def forward(self, atom_embed, fragment_embed, index, mask=None):
        """
           计算原子与片段的对比学习损失

           参数:
               atom_embed: 原子嵌入张量 [batch_size, max_atoms, embed_dim]
               fragment_embed: 片段嵌入张量 [batch_size, max_fragments, embed_dim]
               index: 片段到原子的映射列表，每个元素是一个字典 {片段索引: [原子索引列表]}
               mask: 可选掩码，标识有效位置 [batch_size, max_atoms] 或 [batch_size, max_fragments]

           返回:
               loss: 对比学习损失值
           """
        batch_size = atom_embed.size(0)
        total_loss = 0.0
        valid_fragments_count = 0

        # 遍历批次中的每个分子
        for i in range(batch_size):
            # 获取当前分子的片段到原子映射
            frag_dict = index[i]

            # 遍历当前分子的每个片段
            for frag_idx, atom_indices in frag_dict.items():
                # 获取当前片段的原子嵌入 [num_atoms, embed_dim]
                frag_atoms_embed = atom_embed[i, atom_indices]

                # 计算原子嵌入的平均值 [embed_dim]
                mean_atom_embed = torch.mean(frag_atoms_embed, dim=0)

                # 获取当前片段的嵌入 [embed_dim]
                frag_embed = fragment_embed[i, frag_idx]

                # 计算正样本相似度 (原子平均嵌入与片段嵌入)
                pos_sim = F.cosine_similarity(
                    mean_atom_embed.unsqueeze(0),
                    frag_embed.unsqueeze(0)
                ) / self.temperature

                # 计算负样本相似度 (原子平均嵌入与其他片段嵌入)
                # 获取当前分子所有片段嵌入 [num_fragments, embed_dim]
                all_frags = fragment_embed[i]

                # 创建掩码排除当前片段
                neg_mask = torch.ones(all_frags.size(0), dtype=torch.bool)
                neg_mask[frag_idx] = False

                # 计算负样本相似度
                neg_sims = F.cosine_similarity(
                    mean_atom_embed.unsqueeze(0).expand_as(all_frags),
                    all_frags
                ) / self.temperature

                # 只保留有效的负样本
                neg_sims = neg_sims[neg_mask]

                # 计算对比损失
                numerator = torch.exp(pos_sim)
                denominator = numerator + torch.sum(torch.exp(neg_sims))
                loss = -torch.log(numerator / denominator)

                total_loss += loss
                valid_fragments_count += 1

        # 计算平均损失
        if valid_fragments_count > 0:
            total_loss /= valid_fragments_count
        else:
            total_loss = torch.tensor(0.0, device=atom_embed.device)

        return total_loss

class ContrastiveLoss(nn.Module):
    def __init__(self, loss_computer: str, temperature: float) -> None:
        super().__init__()

        if loss_computer == 'nce_softmax':
            self.loss_computer = NCESoftmaxLoss()
        else:
            raise NotImplementedError(f"Loss Computer {loss_computer} not Support!")
        self.temperature = temperature

    # 假设 当前
    # # 原始视图特征 [batch, dim]
    # z_i = torch.tensor([[1.0, 2.0, 3.0],
    #                     [4.0, 5.0, 6.0]])
    #
    # # 增强视图特征 [batch, dim]
    # z_j = torch.tensor([[1.1, 2.1, 3.1],
    #                     [4.1, 5.1, 6.1]])
    def forward(self, z_i, z_j):
        # SimCSE
        batch_size = z_i.size(0)

        # 拼接后原始特征：
        # [[1.0, 2.0, 3.0],
        #  [4.0, 5.0, 6.0],
        #  [1.1, 2.1, 3.1],
        #  [4.1, 5.1, 6.1]]
        #
        # 归一化后（保留4位小数）：
        # [[0.2673, 0.5345, 0.8018],
        #  [0.4558, 0.5698, 0.6838],
        #  [0.2743, 0.5239, 0.8070],
        #  [0.4564, 0.5705, 0.6846]]
        emb = F.normalize(torch.cat([z_i, z_j]))

        # 相似度矩阵（余弦相似度）：
        # [[1.0000, 0.9746, 0.9999, 0.9745],
        #  [0.9746, 1.0000, 0.9745, 0.9999],
        #  [0.9999, 0.9745, 1.0000, 0.9745],
        #  [0.9745, 0.9999, 0.9745, 1.0000]]
        # 其中，矩阵中任意一个元素都代表 z_i和z_j的相似度。比如 0.9746 代表z_i1 和 z_j2的相似度

        similarity = torch.matmul(emb, emb.t())

        # 处理后相似度矩阵：
        # [[-1e12,  0.9746,  0.9999,  0.9745],
        #  [ 0.9746, -1e12,  0.9745,  0.9999],
        #  [ 0.9999,  0.9745, -1e12,  0.9745],
        #  [ 0.9745,  0.9999,  0.9745, -1e12]]
        # 这个操作的核心目的是消除自相似度对损失计算的影响
        # 对角线元素表示样本与自身的相似度（总是1.0）但在对比学习中，样本自身不应作为负样本
        # 后续Softmax计算时，这些位置的概率趋近于0
        # 这里主要是对 对角线进行处理
        similarity = similarity - torch.eye(batch_size * 2).cuda() * 1e12
        similarity = similarity * 20
        loss = self.loss_computer(similarity)

        return loss

class SupervisedLoss(nn.Module):
    def __init__(self, temperature=0.1, ctr_size='all',
                 ba_temperature=0.07):
        super(SupervisedLoss, self).__init__()
        self.temperature = temperature
        self.ctr_size = ctr_size
        self.ba_temperature = ba_temperature

    #in_feat有三个维度，是个向量
    #第一个维度是 batch_size的大小
    #第二个维度是 每个batch中的分子数量
    #第三个维度是 每个分子的embedding的大小。

    #in_label也有三个维度
    #第一个是batch_size
    #第二个是 每个batch的中该分子对应的标签
    #第三个是 1，因为标签只有1个


    def forward(self, in_feat, in_label=None, mask=None):
        device = (torch.device('cuda')
                  if in_feat.is_cuda
                  else torch.device('cpu'))
        #1、将特征展平
        in_feat = in_feat.view(in_feat.shape[0], in_feat.shape[1], -1)


        #2、计算mask矩阵。mask[i][j] = 1 表示第 i 个样本和第 j 个样本是正对（同类）mask[i][j] = 0 表示负对（不同类）
        #注意：该mask只支持label为一个标签的场景，而不支持多个label！！
        batch_size = in_feat.shape[0]
        if in_label is not None and mask is not None:
            raise ValueError('Cannot define both `in_label` and `mask`')
        elif in_label is None and mask is None:
            #生成单位矩阵
            # mask = [[1, 0, 0],
            #         [0, 1, 0],
            #         [0, 0, 1]]
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif in_label is not None:

            # mask = [[1, 0, 1],  # 样本0与0、2同类
            #         [0, 1, 0],  # 样本1仅与自身同类
            #         [1, 0, 1]]  # 样本2与0、2同类
            in_label = in_label.contiguous().view(-1, 1) # 形状调整为 [B, 1]，B是batch_size
            if in_label.shape[0] != batch_size:
                raise ValueError('Num of in_label does not match num of in_feat')
            mask = torch.eq(in_label, in_label.T).float().to(device)
        else:
            mask = mask.float().to(device)
        #3、假设# Batch Size B=2, 分子数 C=3, 特征维度 D=2
        # in_feat = [
        #     # batch0的3个分子
        #     [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
        #
        #     [[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]
        # ]
        contra_count = in_feat.shape[1]

        # unbind_features = [
        #     [[0.1, 0.2], [0.7, 0.8]],  # 分子0
        #     [[0.3, 0.4], [0.9, 1.0]],  # 分子1
        #     [[0.5, 0.6], [1.1, 1.2]]  # 分子2
        # ]
        #
        # # 拼接后的 contra_feat
        # contra_feat = [
        #     [0.1, 0.2],  # 样本0-分子0
        #     [0.7, 0.8],  # 样本1-分子0
        #     [0.3, 0.4],  # 样本0-分子1
        #     [0.9, 1.0],  # 样本1-分子1
        #     [0.5, 0.6],  # 样本0-分子2
        #     [1.1, 1.2]  # 样本1-分子2
        # ]
        contra_feat = torch.cat(torch.unbind(in_feat, dim=1), dim=0)
        if self.ctr_size == 'one':
            # 只选择一个
            # contra_anchor_feat = [
            #     [0.1, 0.2],  # 样本0-分子0
            #     [0.7, 0.8]  # 样本1-分子0
            # ]
            contra_anchor_feat = in_feat[:, 0]
            contra_anchor_num = 1
        elif self.ctr_size == 'all':
            contra_anchor_feat = contra_feat
            contra_anchor_num = contra_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.ctr_size))

        #4、 anchor_dot_contrast就是contra_anchor_feat的相似度矩阵
        anchor_dot_contrast = torch.div(
            torch.matmul(contra_anchor_feat, contra_feat.T),
            self.temperature)

        #找到每行 最相似的
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        #减去最大值
        logits = anchor_dot_contrast - logits_max.detach()

        #5、扩展mask为两倍，并让mask的对角线元素为0 其余元素不变
        mask = mask.repeat(contra_anchor_num, contra_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * contra_anchor_num).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        #6、
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.ba_temperature) * mean_log_prob_pos
        loss = loss.view(contra_anchor_num, batch_size).mean()

        return loss


