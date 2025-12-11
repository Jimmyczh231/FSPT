from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        # 初始化LogSoftmax层，dim=1表示在第一个维度上进行操作，一般是输入的列，0是行
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        # 将输入张量重塑为(batch_size, num_classes, -1)的形状
        inputs = inputs.view(inputs.size(0), inputs.size(1), -1)

        # 对输入进行LogSoftmax操作
        log_probs = self.logsoftmax(inputs)

        # 将targets转换为one-hot编码的形式
        targets = torch.zeros(inputs.size(0), inputs.size(1)).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)

        # 将targets扩展一个维度
        targets = targets.unsqueeze(-1)

        # 将targets转移到GPU上
        targets = targets.cuda()

        # 计算交叉熵损失
        loss = (- targets * log_probs).mean(0).sum()

        # 返回平均损失
        return loss / inputs.size(2)

