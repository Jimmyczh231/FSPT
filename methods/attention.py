import torch.nn as nn
import torch
import torch.nn.functional as F
import math


def projector(dim, projection_size):
    """
    定义投影头（projection head），用于将输入特征映射到指定的投影空间中。
    """
    return nn.Sequential(
        nn.Linear(dim, dim, bias=False),  # 线性层，输入维度为 dim，输出维度也是 dim。
        nn.ReLU(inplace=True),  # ReLU 激活函数，增加非线性。
        nn.Linear(dim, projection_size, bias=False)  # 线性层，将输出投影到指定的维度 projection_size。
    )


class AttentionSimilarity(nn.Module):
    """
    定义注意力对齐与空间相似性模块（Attention alignment & spatial similarity module）。
    """

    def __init__(self, hidden_size, inner_size=None, drop_prob=0.0, lambda_lse=6., aggregation="mean", bi=False):
        """
        初始化模块，定义特征尺寸、注意力投影等组件。
        """
        super(AttentionSimilarity, self).__init__()

        self.hidden_size = hidden_size  # 隐藏层尺寸（特征的维度）。
        self.inner_size = inner_size if inner_size is not None else hidden_size // 8  # 内部投影尺寸。

        # 定义用于 query、key 和 value 的投影头
        self.query = projector(self.hidden_size, self.inner_size)
        self.key = projector(self.hidden_size, self.inner_size)
        self.value = projector(self.hidden_size, self.inner_size)

        self.dropout = nn.Dropout(drop_prob)  # 定义 Dropout 防止过拟合。

        self.lambda_lse = lambda_lse  # 控制 log-sum-exp 聚合的参数。
        self.aggregation = aggregation  # 定义相似性聚合方式。

        self.bi = bi  # 是否进行双向对齐。

    def contrast_a_with_b(self, query_a, key_a, value_a, query_b, key_b, value_b, features_a, features_b):
        """
        计算支持集特征 features_a 与查询集特征 features_b 之间的对比损失。
        """

        # 1) 空间对齐：扩展 value_a 和 value_b 维度
        value_a = value_a.unsqueeze(0)  # 将 value_a 扩展一个批次维度。
        value_b = value_b.unsqueeze(1)  # 将 value_b 扩展一个批次维度。

        # 使用 query_a 与 key_b 计算注意力得分
        att_scores = torch.matmul(query_a.unsqueeze(0), key_b.unsqueeze(1).transpose(-1, -2).contiguous())
        # 通过 Softmax 归一化得到注意力概率
        att_probs = nn.Softmax(dim=-1)(att_scores / math.sqrt(self.inner_size))
        att_probs = self.dropout(att_probs)  # 应用 Dropout，防止过拟合。

        # 通过注意力概率矩阵与 value_b 计算对齐后的特征
        aligned_features_b = torch.matmul(att_probs, value_b)

        # 确保对齐后的特征尺寸正确
        assert aligned_features_b.size(-1) == self.inner_size
        assert value_b.size(-1) == self.inner_size

        # 2) 计算空间相似性：使用余弦相似度计算 value_a 与 aligned_features_b 的相似性
        similarity = nn.CosineSimilarity(dim=-1)(value_a, aligned_features_b)

        if self.bi:  # 如果启用双向对齐           ###########
            # 使用 query_b 与 key_a 进行反向对齐
            att_scores = torch.matmul(query_b.unsqueeze(1), key_a.unsqueeze(0).transpose(-1, -2).contiguous())
            att_probs = nn.Softmax(dim=-1)(att_scores / math.sqrt(self.inner_size))
            att_probs = self.dropout(att_probs)
            aligned_features_a = torch.matmul(att_probs, value_a)

            # 确保反向对齐后的特征尺寸正确
            assert aligned_features_a.size(-1) == self.inner_size
            assert value_a.size(-1) == self.inner_size

            # 将双向相似度相加
            similarity = similarity + nn.CosineSimilarity(dim=-1)(value_b, aligned_features_a)

        # 3) 根据指定的聚合方式计算总相似度
        if self.aggregation == 'logsum':
            similarity = similarity.mul_(self.lambda_lse).exp_().sum(-1)  # 计算 log-sum-exp 聚合。
            similarity = torch.log(similarity) / self.lambda_lse

        elif self.aggregation == 'mean':
            similarity = similarity.mean(-1)  # 计算平均聚合。

        elif self.aggregation == 'sum':
            similarity = similarity.sum(-1)  # 计算求和聚合。

        elif self.aggregation == 'max':
            similarity = similarity.max(-1)[0]  # 计算最大值聚合。

        return similarity  # 返回相似度结果。

    def forward(self, features_a, features_b):
        """
        前向传播，计算支持集与查询集之间的空间对比损失。
        """

        # 对 features_a 进行投影和转置，使其符合 [Batch, HW, Channels] 的格式
        # features_a = features_a.view(features_a.size(0), features_a.size(1), -1).permute(0, 2, 1).contiguous()          ########
        device = (torch.device('cuda') if features_a.is_cuda else torch.device('cpu'))
        self.to(device)

        # 计算 features_a 的 query, key, value
        query_a = self.query(features_a)
        key_a = self.key(features_a)
        value_a = self.value(features_a)

        # 对 features_b 进行相同的投影和转置
        # features_b = features_b.view(features_b.size(0), features_b.size(1), -1).permute(0, 2, 1).contiguous()

        # 计算 features_b 的 query, key, value
        query_b = self.query(features_b)
        key_b = self.key(features_b)
        value_b = self.value(features_b)

        # 计算支持集与查询集之间的空间对比损失
        spatial_cont_loss = self.contrast_a_with_b(query_a, key_a, value_a, query_b, key_b, value_b, features_a,          ######
                                                   features_b)  # similarity

        return spatial_cont_loss  # 返回空间对比损失结果。 # similarity
