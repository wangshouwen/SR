import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
}


class FixedBatchNorm(nn.BatchNorm2d):
    def forward(self, input):
        return F.batch_norm(input, self.running_mean, self.running_var, self.weight, self.bias,
                            training=False, eps=self.eps)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, BatchNorm=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        # if fix_BN:
        #     BatchNorm = FixedBatchNorm
        # else:
        #     BatchNorm = nn.BatchNorm2d
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, strides=(2, 2, 2, 2), frozen_stage=-1, fix_BN=False):
        self.inplanes = 64
        super(ResNet, self).__init__()

        if fix_BN:
            BatchNorm = FixedBatchNorm
        else:
            BatchNorm = nn.BatchNorm2d

        self.frozen_stage = frozen_stage
        if self.frozen_stage > -1:
            for i in range(0, self.frozen_stage + 1):
                setattr(self, f'BN_stage{i}', FixedBatchNorm)
            for i in range(self.frozen_stage + 1, 5):
                setattr(self, f'BN_stage{i}', BatchNorm)
        else:
            for i in range(0, 5):
                setattr(self, f'BN_stage{i}', BatchNorm)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=strides[0], padding=3, bias=False)
        self.bn1 = self.BN_stage0(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1, BatchNorm=self.BN_stage1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], BatchNorm=self.BN_stage2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], BatchNorm=self.BN_stage3)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], BatchNorm=self.BN_stage4)

        if self.frozen_stage > -1:
            for m in [self.conv1, self.bn1]:
                for param in m.parameters():
                    param.requires_grad = False
            for i in range(1, self.frozen_stage + 1):
                m = getattr(self, f'layer{i}')
                for param in m.parameters():
                    param.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, BatchNorm=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, BatchNorm)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet50(pretrained=True, **kwargs):

    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet50'])
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        model.load_state_dict(state_dict)
    return model


def resnet101(pretrained=True, **kwargs):

    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet101'])
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        model.load_state_dict(state_dict)
    return model
