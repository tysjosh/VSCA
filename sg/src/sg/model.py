"""CIFAR-style ResNet-18 with auxiliary GAP+Linear heads at stage outputs (Table I)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

STAGE_CHANNELS = {"s1": 64, "s2": 128, "s3": 256, "s4": 512}
BLOCK_STAGE = [1, 1, 2, 2, 3, 3, 4, 4]


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


def _gap(h: torch.Tensor) -> torch.Tensor:
    return h.mean(dim=(2, 3))


class ResNet18(nn.Module):
    """3x3 stem, no max-pool; stages s1..s4 = 64/128/256/512 channels, strides 1/2/2/2."""

    def __init__(self, n_classes: int = 100, aux_stages=(), n_aux: int = 20):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        layers, cin = [], 64
        for cout, stride in ((64, 1), (128, 2), (256, 2), (512, 2)):
            layers.append(nn.Sequential(BasicBlock(cin, cout, stride), BasicBlock(cout, cout, 1)))
            cin = cout
        self.layer1, self.layer2, self.layer3, self.layer4 = layers
        self.fc = nn.Linear(512, n_classes)
        self.aux_stages = tuple(aux_stages)
        self.heads = nn.ModuleDict({s: nn.Linear(STAGE_CHANNELS[s], n_aux) for s in self.aux_stages})

    @property
    def layers(self):
        return (self.layer1, self.layer2, self.layer3, self.layer4)

    def forward(self, x, return_stages: bool = False):
        h = self.stem(x)
        stages = {}
        for i, layer in enumerate(self.layers):
            h = layer(h)
            stages[f"s{i + 1}"] = h
        logits = self.fc(_gap(h))
        aux = {s: head(_gap(stages[s])) for s, head in self.heads.items()}
        return logits, aux, (stages if return_stages else None)

    def block_features(self, x) -> list:
        """GAP of each of the 8 BasicBlock outputs, [B, C] each."""
        h = self.stem(x)
        feats = []
        for layer in self.layers:
            for block in layer:
                h = block(h)
                feats.append(_gap(h))
        return feats

    def features(self, x) -> torch.Tensor:
        h = self.stem(x)
        for layer in self.layers:
            h = layer(h)
        return _gap(h)
