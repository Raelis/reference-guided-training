import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()

        self.in_planes = 16


        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)


        self.layer1 = self._make_layer(block, 16,  num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32,  num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64,  num_blocks[2], stride=2)

        self.linear = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []

        layers.append(block(self.in_planes, planes, stride))
        self.in_planes = planes * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, 1))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = F.adaptive_avg_pool2d(out, 1)
        out = torch.flatten(out, 1)

        return self.linear(out)


class MyModel(nn.Module):
    model_name = "ResNet20"

    def __init__(self, num_features=1024, output_dim=10):
        super().__init__()
        self.model = ResNet(BasicBlock, [3, 3, 3], num_classes=output_dim)

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def prepare_trial(trial):
        return {}

    @staticmethod
    def prepare_model_params(model_params):
        other_params = {}

        for k in ["batch_size", "window_size", "lr", "weight_decay",
                  "lamb", "delta", "margin"]:
            if k in model_params:
                other_params[k] = model_params[k]
                model_params.pop(k, None)

        return {}, other_params

    @staticmethod
    def model_postprocess(model):
        pass
