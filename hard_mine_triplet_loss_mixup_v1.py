from __future__ import absolute_import
from __future__ import division

import sys
import torch
import torch.nn as nn

# random permutation between ids in a batch

class TripletLoss_Mixup(nn.Module):

    def __init__(self, margin=0.3):
        super(TripletLoss_Mixup, self).__init__()
        self.margin = margin
        # self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.ranking_loss_1 = nn.MarginRankingLoss(margin=0.3)
        self.ranking_loss_2 = nn.MarginRankingLoss(margin=0.3)
        self.ranking_loss_3 = nn.MarginRankingLoss(margin=0.3)

    def forward(self, inputs, targets_a, targets_b, lam, epoch):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask_a = targets_a.expand(n, n).eq(targets_a.expand(n, n).t())
        mask_b = targets_b.expand(n, n).eq(targets_b.expand(n, n).t())

        # mask_same_a_and_b = mask_a & mask_b & (~torch.eye(n, dtype=torch.uint8, device='cuda'))
        mask_same_a_and_b = mask_a & mask_b
        mask_same_a = mask_a & (~mask_b)
        mask_same_b = (~mask_a) & mask_b
        mask_same_a_or_b = mask_same_a | mask_same_b
        mask_dif_a_and_b = (~mask_a) & (~mask_b)

        dist_same_a_and_b_hp, dist_dif_a_and_b_hn = [], []
        dist_same_a_or_b_sp, dist_same_a_or_b_sn = [], []

        for i in range(n):

            dist_same_a_and_b_hp.append(dist[i][mask_same_a_and_b[i]].max().unsqueeze(0))
            # dist_same_a_and_b_hp.append(dist[i][mask_same_a_and_b[i]].min().unsqueeze(0))
            dist_dif_a_and_b_hn.append(dist[i][mask_dif_a_and_b[i]].min().unsqueeze(0))
            if mask_same_a_or_b[i].any() and epoch > 50:
            # if mask_same_a_or_b[i].any():
                dist_same_a_or_b_sp.append(dist[i][mask_same_a_or_b[i]].max().unsqueeze(0))
                # dist_same_a_or_b_sp.append(dist[i][mask_same_a_or_b[i]].min().unsqueeze(0))
                dist_same_a_or_b_sn.append(dist[i][mask_same_a_or_b[i]].min().unsqueeze(0))
            else:
                dist_same_a_or_b_sp.append(dist[i][mask_same_a_and_b[i]].max().unsqueeze(0))
                dist_same_a_or_b_sn.append(dist[i][mask_dif_a_and_b[i]].min().unsqueeze(0))

        dist_same_a_and_b_hp = torch.cat(dist_same_a_and_b_hp)
        dist_dif_a_and_b_hn = torch.cat(dist_dif_a_and_b_hn)

        dist_same_a_or_b_sp = torch.cat(dist_same_a_or_b_sp)
        dist_same_a_or_b_sn = torch.cat(dist_same_a_or_b_sn)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_same_a_and_b_hp)

        loss_1 = self.ranking_loss_1(dist_dif_a_and_b_hn, dist_same_a_and_b_hp, y)
        loss_2 = self.ranking_loss_2(dist_dif_a_and_b_hn, dist_same_a_or_b_sp, y)
        loss_3 = self.ranking_loss_3(dist_same_a_or_b_sn, dist_same_a_and_b_hp, y)
        loss = loss_1 + loss_2 + loss_3

        # return self.ranking_loss(dist_dif_a_and_b_hn, dist_same_a_and_b_hp, y)
        return loss
