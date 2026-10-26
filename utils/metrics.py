import torch
import numpy as np


class AverageValueMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def add(self, value, n=1):
        self.val = value
        self.sum += value * n
        self.n += n
        self.mean = self.sum / self.n

    def value(self):
        return self.val

    def average(self):
        return self.mean

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.val = 0.0
        self.mean = 0.0

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class SumValueMeter(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def add(self, pos_num, neg_num):
        self.pos_num += pos_num
        self.neg_num += neg_num

    def value(self):
        return self.pos_num, self.neg_num

    def reset(self):
        self.pos_num = torch.zeros(self.num_classes)
        self.neg_num = torch.zeros(self.num_classes)


class AveragePrecisionMeter(object):
    """Average precision per class for multi-label classification.

    Accumulates an `N x K` score matrix and an `N x K` target matrix, where a
    target is `1` for a positive label, `-1` for a negative one and `0` for an
    unknown (unannotated) one. With ``difficult_examples=True`` the unknown
    entries are skipped when computing AP.
    """

    def __init__(self, difficult_examples=True):
        super(AveragePrecisionMeter, self).__init__()
        self.difficult_examples = difficult_examples
        self.reset()

    def reset(self):
        self._scores = []
        self._targets = []
        self._cache = None

    def add(self, output, target):
        """Append a batch of scores and targets.

        Args:
            output (Tensor): `N x K` predicted score per class.
            target (Tensor): `N x K` labels in {-1, 0, 1}.
        """
        if not torch.is_tensor(output):
            output = torch.from_numpy(output)
        if not torch.is_tensor(target):
            target = torch.from_numpy(target)

        output = output.view(-1, 1) if output.dim() == 1 else output
        target = target.view(-1, 1) if target.dim() == 1 else target
        assert output.dim() == 2, 'Wrong output size (should be 1D or 2D with one column per class)'
        assert target.dim() == 2, 'Wrong target size (should be 1D or 2D with one column per class)'
        assert output.shape == target.shape, 'output and target must have the same shape'
        if self._scores:
            assert target.size(1) == self._scores[0].size(1), \
                'Dimensions for output should match previously added examples'

        self._scores.append(output.detach().float().cpu())
        self._targets.append(target.detach().float().cpu())
        self._cache = None

    @property
    def scores(self):
        """All accumulated scores as one `N x K` tensor."""
        if self._cache is None:
            if not self._scores:
                self._cache = (torch.empty(0), torch.empty(0))
            else:
                self._cache = (torch.cat(self._scores, 0), torch.cat(self._targets, 0))
        return self._cache[0]

    @property
    def targets(self):
        """All accumulated targets as one `N x K` tensor."""
        _ = self.scores
        return self._cache[1]

    def value(self):
        """Average precision for each class, as a length-K tensor."""
        scores, targets = self.scores, self.targets
        if scores.numel() == 0:
            return 0

        ap = torch.zeros(scores.size(1))
        for k in range(scores.size(1)):
            ap[k] = AveragePrecisionMeter.average_precision(
                scores[:, k], targets[:, k], self.difficult_examples)
        return ap

    @staticmethod
    def average_precision(output, target, difficult_examples=True):
        # Rank by score, then walk down the ranking accumulating precision@i at
        # every positive. Unknown labels (target == 0) are skipped entirely.
        order = torch.argsort(output, descending=True)
        ranked = target[order]
        if difficult_examples:
            ranked = ranked[ranked != 0]

        pos = (ranked == 1).double()
        num_pos = pos.sum()
        if num_pos == 0:
            return 0.0

        hits = pos.cumsum(0)
        seen = torch.arange(1, ranked.numel() + 1, dtype=torch.double)
        return float((pos * hits / seen).sum() / num_pos)

    def overall(self):
        if self.scores.numel() == 0:
            return 0
        return self.evaluation(self.scores.numpy(), self.targets.numpy())

    def overall_topk(self, k):
        scores, targets = self.scores, self.targets
        if scores.numel() == 0:
            return 0

        # Keep only the k highest-scoring classes per sample; their sign is what
        # `evaluation` thresholds on.
        topk = scores.topk(k, dim=1).indices
        pred = torch.full_like(scores, -1.0)
        pred.scatter_(1, topk, torch.gather(scores, 1, topk).sign())
        # sign() maps an exact 0 to 0, which would count as a positive below.
        pred[pred == 0] = 1.0
        return self.evaluation(pred.numpy(), targets.numpy())

    def evaluation(self, scores_, targets_):
        """Overall/per-class precision, recall and F1.

        Both arguments are treated as read-only; unknown labels (0) and negative
        labels (-1) are both counted as negatives.
        """
        targets_ = np.where(targets_ == 1, 1, 0)
        preds = scores_ >= 0

        Ng = targets_.sum(axis=0)                 # ground-truth positives per class
        Np = preds.sum(axis=0)                    # predicted positives per class
        Nc = (targets_ * preds).sum(axis=0)       # correct positives per class

        Np = np.where(Np == 0, 1, Np)
        Ng_safe = np.where(Ng == 0, 1, Ng)

        OP = Nc.sum() / Np.sum()
        OR = Nc.sum() / max(Ng.sum(), 1)
        OF1 = (2 * OP * OR) / (OP + OR) if (OP + OR) > 0 else 0.0

        classNum = scores_.shape[1]
        CP = (Nc / Np).sum() / classNum
        CR = (Nc / Ng_safe).sum() / classNum
        CF1 = (2 * CP * CR) / (CP + CR) if (CP + CR) > 0 else 0.0

        return OP, OR, OF1, CP, CR, CF1
