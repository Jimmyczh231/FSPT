import torch
import torch.nn as nn
import torch.nn.functional as F



cos = nn.CosineSimilarity(dim=1, eps=1e-6)


def Xcos(ftrain, ftest, ytrain, batch_size, num_train, num_test, K):
    ftrain = ftrain.view(batch_size, num_train, -1)
    prototypes = torch.bmm(ytrain, ftrain)
    prototypes = prototypes.view(batch_size, -1, *ftest.size()[1:])
    ftest = ftest.view(batch_size, num_test, -1, 768)
    prototypes = prototypes.unsqueeze(1).repeat(1, num_test, 1, 1, 1)
    ftest = ftest.unsqueeze(2).repeat(1, 1, K, 1, 1)
    prototypes = Long_alignment(prototypes, ftest)

    B, n2, n1, N, C = prototypes.size()

    prototypes = prototypes.view(-1, prototypes.size(3), 768)
    ftest = ftest.view((-1, ftest.size(3), 768))

    prototypes = prototypes.view(-1, 768)
    ftest = ftest.view(-1, 768)
    similarity = 10 * cos(prototypes, ftest).view(B * n2, n1, -1)
    return similarity


def Long_alignment(support_x, query_x):
    support_x = F.normalize(support_x, p=2, dim=-1, eps=1e-12)
    query_x = F.normalize(query_x, p=2, dim=-1, eps=1e-12)
    support_x = support_x.transpose(3, 4)

    Mt = torch.matmul(query_x, support_x)

    Mt = F.softmax(Mt, dim=4)

    support_x = support_x.transpose(3, 4)

    align_support = torch.matmul(Mt, support_x)

    return align_support