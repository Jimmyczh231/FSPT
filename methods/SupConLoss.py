import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from methods.attention import AttentionSimilarity
class SupConLoss(nn.Module):


    def __init__(self, temperature=0.1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def _compute_logits(self, features_a, features_b, attention):
        # global similarity
        if features_a.dim() == 2:
            features_a = F.normalize(features_a, dim=1, p=2)
            features_b = F.normalize(features_b, dim=1, p=2)
            contrast = torch.matmul(features_a, features_b.T)

        # spatial similarity
        elif features_a.dim() == 3:
            contrast = attention(features_a, features_b)

        else:
            raise ValueError

        contrast = torch.div(contrast, self.temperature)
        return contrast

    def forward(self, features_a,  features_b=None, labels=None, attention=None):
        device = (torch.device('cuda') if features_a.is_cuda else torch.device('cpu'))
        num_features, num_labels = features_a.shape[0], labels.shape[0]
        if features_b is None:
            features_b = features_a
            logits_mask = (1. - torch.eye(num_features)).to(device)
        else:
            logits_mask = torch.ones(num_features, num_features).to(device)

        if labels is None:
            mask = torch.eye(num_labels, dtype=torch.float32).to(device)
        else:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)

        if num_features != num_labels:
            assert num_labels * 2 == num_features
            mask = mask.repeat(2, 2)

        contrast = self._compute_logits(features_a, features_b, attention)

        mask = mask * logits_mask

        normalization = mask.sum(1)
        normalization[normalization == 0] = 1.

        logits_max, _ = torch.max(contrast, dim=1, keepdim=True)
        logits = contrast - logits_max.detach()
        exp_logits = torch.exp(logits)

        exp_logits = exp_logits * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / normalization
        loss = -mean_log_prob_pos.mean()

        return loss * self.temperature
