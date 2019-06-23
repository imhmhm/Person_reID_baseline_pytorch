from __future__ import absolute_import
from __future__ import division

import sys
import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    """
    contrastive loss applied to each element in a batch

    """

    def __init__(self, margin=0.3):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        # self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.hinge_loss = nn.HingeEmbeddingLoss(margin=margin, reduction='sum')

    def forward(self, inputs, targets):

        n = inputs.size(0)

        # Compute pairwise distance
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        mask_same = mask & (~torch.eye(n, dtype=torch.uint8, device='cuda'))

        dist_p, dist_n = [], []
        for i in range(n):

            dist_p.extend(dist[i][mask_same[i]])
            dist_n.extend(dist[i][mask[i] == 0])

        dist_p = torch.stack(dist_p)
        dist_n = torch.stack(dist_n)
        dist = torch.cat([dist_p, dist_n])

        # Compute ranking hinge loss
        y1 = torch.ones_like(dist_p)
        y2 = -1 * torch.ones_like(dist_n)
        y = torch.cat([y1, y2])

        loss = 0.5 * self.hinge_loss(dist, y)
        return loss

# hard mining
class HardContrastiveLoss(nn.Module):
    """
    mining hard examples in batches
    """
    def __init__(self, margin=0.3):
        super(HardContrastiveLoss, self).__init__()
        self.margin = margin
        self.hinge_loss = nn.HingeEmbeddingLoss(margin=margin,  reduction='mean')

    def forward(self, inputs, targets):

        n = inputs.size(0)

        # Compute pairwise distance
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        mask_same = mask & (~torch.eye(n, dtype=torch.uint8, device='cuda'))

        dist_p, dist_n = [], []
        for i in range(n):

            dist_p.append(dist[i][mask_same[i]].max().unsqueeze(0))
            dist_n.append(dist[i][mask[i] == 0].min().unsqueeze(0))

        dist_p = torch.cat(dist_p)
        dist_n = torch.cat(dist_n)
        dist = torch.cat([dist_p, dist_n])

        # Compute ranking hinge loss
        y1 = torch.ones_like(dist_p)
        y2 = -1 * torch.ones_like(dist_n)
        y = torch.cat([y1, y2])

        loss = self.hinge_loss(dist, y)

        return loss
