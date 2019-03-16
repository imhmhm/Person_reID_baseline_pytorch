from __future__ import absolute_import
from __future__ import division

import sys
import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss_1 = nn.MarginRankingLoss(margin=0.8)
        self.ranking_loss_2 = nn.MarginRankingLoss(margin=0.5)
        self.ranking_loss_3 = nn.MarginRankingLoss(margin=0.3)

    def forward(self, inputs, targets, flags, epoch):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask_1 = targets.expand(n, n).eq(targets.expand(n, n).t())
        mask_2 = flags.expand(n, n).eq(flags.expand(n, n).t())
        mask_same_hp = mask_1 & mask_2
        mask_same_hn = (~mask_1) & mask_2
        mask_dif_hp = mask_1 & (~mask_2)
        mask_dif_hn = (~mask_1) & (~mask_2)

        dist_same_ap, dist_same_an = [], []
        dist_dif_ap, dist_dif_an = [], []
        dist_dif_neg = []
        # print('############################################')
        for i in range(n):
            dist_same_ap.append(dist[i][mask_same_hp[i]].max().unsqueeze(0))
            dist_same_an.append(dist[i][mask_same_hn[i]].min().unsqueeze(0))
            dist_dif_ap.append(dist[i][mask_dif_hp[i]].max().unsqueeze(0))
            dist_dif_an.append(dist[i][mask_dif_hn[i]].min().unsqueeze(0))
            same_min_idx = (dist[i] == (dist[i][mask_same_hn[i]].min())).nonzero().squeeze(1)[0]
            dif_min_idx = (dist[i] == (dist[i][mask_dif_hn[i]].min())).nonzero().squeeze(1)[0]
            # print(dist[i][dif_min_idx])
            # if len(dif_min_idx.size()) != 1:
            #     print(dist[i][dif_min_idx])
            #     sys.exit()
            dist_dif_neg.append(dist[same_min_idx][dif_min_idx].unsqueeze(0))


            # print(dist_same_ap, dist_same_an, dist_dif_ap, dist_dif_an, same_min_idx, dif_min_idx, dist_dif_neg)
            # sys.exit()
        dist_same_ap = torch.cat(dist_same_ap)
        dist_same_an = torch.cat(dist_same_an)
        dist_dif_ap = torch.cat(dist_dif_ap)
        dist_dif_an = torch.cat(dist_dif_an)
        dist_dif_neg = torch.cat(dist_dif_neg)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_same_ap)
        loss_same = self.ranking_loss_1(dist_same_an, dist_same_ap, y)
        loss_dif = self.ranking_loss_2(dist_dif_an, dist_dif_ap, y)
        loss_neg = self.ranking_loss_3(dist_dif_neg, dist_same_ap, y)
        loss = loss_same + loss_dif + loss_neg
        return loss
