"""PreActResNet-18 (He et al., 2016) as used by RAMP / E-AT on CIFAR."""
import torch
import torch.nn as nn
import torch.nn.functional as F

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


class Normalize(nn.Module):
    """Normalisation lives inside the model so every attack operates in [0, 1] pixel space
    and epsilons are directly comparable with RobustBench / AutoAttack numbers."""

    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std


def _act(name):
    if name == "relu":
        return F.relu
    beta = int(name.replace("softplus", ""))  # "softplus1" as in the RAMP / E-AT model zoo
    return lambda t: F.softplus(t, beta=beta)


class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, activation="relu"):
        super().__init__()
        self.act = _act(activation)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.shortcut = None
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride, bias=False))

    def forward(self, x):
        out = self.act(self.bn1(x))
        sc = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.conv2(self.act(self.bn2(out)))
        return out + sc


class PreActResNet(nn.Module):
    def __init__(self, num_blocks=(2, 2, 2, 2), num_classes=10, activation="relu"):
        super().__init__()
        self.in_planes, self.activation = 64, activation
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.layer1 = self._make_layer(64, num_blocks[0], 1)
        self.layer2 = self._make_layer(128, num_blocks[1], 2)
        self.layer3 = self._make_layer(256, num_blocks[2], 2)
        self.layer4 = self._make_layer(512, num_blocks[3], 2)
        self.bn = nn.BatchNorm2d(512)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes, n, stride):
        layers = []
        for s in [stride] + [1] * (n - 1):
            layers.append(PreActBlock(self.in_planes, planes, s, self.activation))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(out)


def build_model(name: str = "preactresnet18", dataset: str = "cifar10") -> nn.Module:
    """preactresnet18          ReLU, CIFAR normalisation inside the model
    preactresnet18_softplus  the RAMP / E-AT architecture (softplus(beta=1) in the blocks, no input
                             normalisation); loads the official pretr_{Linf,L1,L2}.pth checkpoints"""
    num_classes = 100 if dataset == "cifar100" else 10
    mean, std = (CIFAR100_MEAN, CIFAR100_STD) if dataset == "cifar100" else (CIFAR10_MEAN, CIFAR10_STD)
    if name == "preactresnet18":
        return nn.Sequential(Normalize(mean, std), PreActResNet((2, 2, 2, 2), num_classes))
    if name == "preactresnet18_softplus":
        return nn.Sequential(nn.Identity(), PreActResNet((2, 2, 2, 2), num_classes, activation="softplus1"))
    raise ValueError(f"unknown model {name}")


def load_weights(model: nn.Module, path: str, map_location="cpu"):
    """Accepts our checkpoints ({'model': sd}) and raw state_dicts from the RAMP / E-AT repos
    (keys without the leading '1.' of our Sequential wrapper)."""
    sd = torch.load(path, map_location=map_location)
    sd = sd.get("model", sd.get("state_dict", sd))
    if not any(k.startswith("1.") for k in sd):
        sd = {f"1.{k}": v for k, v in sd.items()}
    own = model.state_dict()
    sd.update({k: v for k, v in own.items() if k.startswith("0.") and k not in sd})  # Normalize buffers
    model.load_state_dict(sd)
    return model
