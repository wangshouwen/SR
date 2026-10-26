import torch
import torch.nn as nn
from .resnet import resnet101, resnet50


class SR_ResNet(nn.Module):
    def __init__(self, num_classes=20, depth=101, input_dim=2048, frozen_stage=-1,
                 fix_BN=False, topk=5, alpha=0.5, noSR=False, pretrained=True):
        super(SR_ResNet, self).__init__()

        self.num_classes = num_classes
        self.input_dim = input_dim
        self.frozen_stage = frozen_stage  # values: -1, 0, 1, 2, 3, 4
        self.prototype = None
        self.topk = topk
        self.alpha = alpha
        self.noSR = noSR

        if depth == 101:
            backbone = resnet101
        else:
            backbone = resnet50

        # pretrained=False skips the ImageNet download when the weights are about
        # to be overwritten by a checkpoint anyway (see predict.py).
        self.backbone = backbone(pretrained=pretrained, frozen_stage=self.frozen_stage, fix_BN=fix_BN)
        self.classifier = nn.Conv2d(self.input_dim, self.num_classes, 1, bias=False)
        self.add_modules = nn.ModuleList([self.classifier])

    def forward(self, x, divide=False, visualize=False):
        x = self.backbone(x)  # [bs, 2048, 14, 14] for imgsize 448
        cam = self.classifier(x)  # [bs, num_classes, 14, 14]
        y_raw = cam.flatten(2)  # # [bs, num_classes, 196]
        y_avg = torch.mean(y_raw, dim=2)
        y_sort = torch.sort(y_raw, dim=2, descending=True)[0]
        y_max_avg = y_sort[:, :, :self.topk].mean(2)

        if self.noSR:
            score = y_avg
        else:
            score = y_avg + self.alpha * y_max_avg

        if visualize:
            return score, cam

        if divide:
            return score, y_avg

        return score

    def trainable_parameters(self):
        backbone_group = []
        stage4_group = []

        first_trainable = self.frozen_stage
        if first_trainable == -1:
            for m in [self.backbone.conv1, self.backbone.bn1]:
                backbone_group.extend(m.parameters())
            first_trainable = 0

        for i in range(first_trainable + 1, 4):
            m = getattr(self.backbone, f'layer{i}')
            params = list(m.parameters())
            backbone_group.extend(params)

        if self.frozen_stage < 4:
            param1 = list(self.backbone.layer4.parameters())
            stage4_group.extend(param1)

        add_group = list(self.add_modules.parameters())

        return backbone_group, stage4_group, add_group
