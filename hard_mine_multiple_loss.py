from __future__ import absolute_import
from __future__ import division

import sys
import torch
import torch.nn as nn


class HardQuadLoss(nn.Module):

    def __init__(self, margin=0.3):
        super(HardQuadLoss, self).__init__()
        self.margin = margin
        self.ranking_loss_1 = nn.MarginRankingLoss(margin=1.2)
        self.ranking_loss_2 = nn.MarginRankingLoss(margin=0.3)
        # self.ranking_loss_3 = nn.MarginRankingLoss(margin=0.3)

    def forward(self, inputs, targets):

        n = inputs.size(0)

        # Compute pairwise distance
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask_1 = targets.expand(n, n).eq(targets.expand(n, n).t())
        mask_same_hp = mask_1
        mask_same_hn = (~mask_1)

        dist_same_ap, dist_same_an = [], []
        dist_dif_neg = []
        # print('############################################')
        for i in range(n):
            dist_same_ap.append(dist[i][mask_same_hp[i]].max().unsqueeze(0))
            dist_same_an.append(dist[i][mask_same_hn[i]].min().unsqueeze(0))

            same_min_idx = (dist[i] == (dist[i][mask_same_hn[i]].min())).nonzero().squeeze(1)[0]

            dist_dif_neg.append(dist[same_min_idx][mask_same_hn[same_min_idx]].min().unsqueeze(0))

        dist_same_ap = torch.cat(dist_same_ap)
        dist_same_an = torch.cat(dist_same_an)

        dist_dif_neg = torch.cat(dist_dif_neg)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_same_ap)
        loss_same = self.ranking_loss_1(dist_same_an, dist_same_ap, y)
        loss_dif = self.ranking_loss_2(dist_dif_neg, dist_same_ap, y)
        loss = loss_same + loss_dif

        return loss
