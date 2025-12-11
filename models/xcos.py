import torch
import torch.nn as nn
import torch.nn.functional as F



cos = nn.CosineSimilarity(dim=1, eps=1e-6)  # 定义余弦相似度计算器


def Xcos(ftrain, ftest, ytrain, batch_size, num_train, num_test, K):
    ftrain = ftrain.view(batch_size, num_train, -1)  # 调整训练集特征张量的形状
    prototypes = torch.bmm(ytrain, ftrain)
    # prototypes = prototypes.view(batch_size, num_train, -1, 768)
    prototypes = prototypes.view(batch_size, -1, *ftest.size()[1:])
    ftest = ftest.view(batch_size, num_test, -1, 768)
    prototypes = prototypes.unsqueeze(1).repeat(1, num_test, 1, 1, 1)  # 将原型向量扩展为与测试集特征相同的形状
    ftest = ftest.unsqueeze(2).repeat(1, 1, K, 1, 1)  # 将测试集特征扩展为与原型向量相同的形状
    # prototypes = prototypes.view(num_train, -1, 768)
    prototypes = Long_alignment(prototypes, ftest)  # 对齐

    B, n2, n1, N, C = prototypes.size()  # 获取输入张量的大小

    prototypes = prototypes.view(-1, prototypes.size(3), 768)
    ftest = ftest.view((-1, ftest.size(3), 768))

    prototypes = prototypes.view(-1, 768)
    ftest = ftest.view(-1, 768)
    similarity = 10 * cos(prototypes, ftest).view(B * n2, n1, -1)
    return similarity

def Long_alignment(support_x, query_x):
    # b, q, s, c, h, w = support_x.shape  # 获取张量形状信息
    support_x = F.normalize(support_x, p=2, dim=-1, eps=1e-12)  # 对支持集张量进行L2范数归一化
    query_x = F.normalize(query_x, p=2, dim=-1, eps=1e-12)  # 对查询集张量进行L2范数归一化
    # support_x = support_x.view(b, q, s, c, h * w)  # 调整支持集张量形状
    support_x = support_x.transpose(3, 4)  # 调整查询集张量形状并转置

    Mt = torch.matmul(query_x, support_x)  # 计算点积得到相似度矩阵

    Mt = F.softmax(Mt, dim=4)  # 对相似度矩阵进行softmax归一化

    support_x = support_x.transpose(3, 4)  # 调整支持集张量形状

    align_support = torch.matmul(Mt, support_x)  # 计算对齐的支持集张量
    # align_support = align_support.transpose(3, 4)  # 调整对齐的支持集张量形状

    return align_support  # 返回对齐的支持集张量

