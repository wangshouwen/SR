import torch
import torch.nn as nn


class BCELoss(nn.Module):
    def __init__(self, reduce=None, size_average=None):
        super(BCELoss, self).__init__()
        self.reduce = reduce
        self.size_average = size_average
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, x, y):
        weight = torch.abs(torch.clamp(y, -1, 1))
        y = torch.clamp(y, 0, 1)
        loss = self.BCEWithLogitsLoss(x, y)
        loss = loss * weight

        if self.reduce:
            if self.size_average:
                return torch.mean(loss[weight > 0]) if torch.sum(weight) != 0 else torch.mean(loss)
            return torch.sum(loss)
        return loss
