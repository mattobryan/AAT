"""PreActResNet-18 (He et al., 2016) as used by RAMP / E-AT on CIFAR."""
import re

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


class GatedBN(nn.Module):
    """BatchNorm (shared statistics, no affine) followed by a per-sample, gate-weighted affine:
    gamma_eff = g @ gamma, beta_eff = g @ beta, with g (N, K) a softmax gate set on `self.g` by the parent
    model before the blocks run. K = number of threat types."""

    def __init__(self, c, K):
        super().__init__()
        self.bn = nn.BatchNorm2d(c, affine=False)
        self.gamma = nn.Parameter(torch.ones(K, c))
        self.beta = nn.Parameter(torch.zeros(K, c))
        self.g = None

    def forward(self, x):
        assert self.g is not None, "GatedBN.g must be set by the parent model before forward"
        g = self.g.to(x.dtype)
        gamma, beta = g @ self.gamma.to(x.dtype), g @ self.beta.to(x.dtype)
        return self.bn(x) * gamma[:, :, None, None] + beta[:, :, None, None]


def _norm_layer(gated_K):
    return nn.BatchNorm2d if not gated_K else (lambda c: GatedBN(c, gated_K))


class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, activation="relu", gated_K=0):
        super().__init__()
        self.act = _act(activation)
        norm = _norm_layer(gated_K)
        self.bn1 = norm(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn2 = norm(planes)
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
    def __init__(self, num_blocks=(2, 2, 2, 2), num_classes=10, activation="relu", gated_K=0):
        """gated_K > 0: "squint" model. A gate head on the stem features predicts a softmax over K threat types
        from the (possibly perturbed) input, and every BatchNorm becomes a GatedBN conditioned on it."""
        super().__init__()
        self.in_planes, self.activation, self.gated_K = 64, activation, gated_K
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.layer1 = self._make_layer(64, num_blocks[0], 1)
        self.layer2 = self._make_layer(128, num_blocks[1], 2)
        self.layer3 = self._make_layer(256, num_blocks[2], 2)
        self.layer4 = self._make_layer(512, num_blocks[3], 2)
        self.bn = _norm_layer(gated_K)(512)
        self.linear = nn.Linear(512, num_classes)
        self.last_gate_logits = None
        if gated_K:
            self.gate_conv = nn.Conv2d(64, 32, 3, 2, 1, bias=False)
            self.gate_bn = nn.BatchNorm2d(32)
            self.gate_fc = nn.Linear(32, gated_K)

    def _make_layer(self, planes, n, stride):
        layers = []
        for s in [stride] + [1] * (n - 1):
            layers.append(PreActBlock(self.in_planes, planes, s, self.activation, self.gated_K))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        if self.gated_K:
            h = F.adaptive_avg_pool2d(F.relu(self.gate_bn(self.gate_conv(out))), 1).flatten(1)
            self.last_gate_logits = self.gate_fc(h)  # (N, K), raw logits; differentiable w.r.t. the input
            g = F.softmax(self.last_gate_logits, dim=1)
            for m in self.modules():
                if isinstance(m, GatedBN):
                    m.g = g
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(out)


def build_model(name: str = "preactresnet18", dataset: str = "cifar10") -> nn.Module:
    """preactresnet18          ReLU, CIFAR normalisation inside the model
    preactresnet18_softplus  the RAMP / E-AT architecture (softplus(beta=1) in the blocks, no input
                             normalisation); loads the official pretr_{Linf,L1,L2}.pth checkpoints
    preactresnet18_gated<K>  input-conditioned "squint" model: gate head + K-way gated BatchNorm affines"""
    num_classes = 100 if dataset == "cifar100" else 10
    mean, std = (CIFAR100_MEAN, CIFAR100_STD) if dataset == "cifar100" else (CIFAR10_MEAN, CIFAR10_STD)
    if name == "preactresnet18":
        return nn.Sequential(Normalize(mean, std), PreActResNet((2, 2, 2, 2), num_classes))
    if name == "preactresnet18_softplus":
        return nn.Sequential(nn.Identity(), PreActResNet((2, 2, 2, 2), num_classes, activation="softplus1"))
    mg = re.fullmatch(r"preactresnet18_gated(\d+)", name)
    if mg:
        return nn.Sequential(Normalize(mean, std), PreActResNet((2, 2, 2, 2), num_classes, gated_K=int(mg.group(1))))
    if name.startswith("rb:"):
        # RobustBench architecture by model-zoo id (e.g. rb:Gowal2020Uncovering_28_10_extra). Built WITHOUT
        # downloading its weights; normalisation is inside the RobustBench model, so inputs stay in [0, 1].
        from robustbench.model_zoo import model_dicts
        from robustbench.data import BenchmarkDataset
        from robustbench.model_zoo.enums import ThreatModel
        model = model_dicts[BenchmarkDataset(dataset)][ThreatModel.Linf][name[3:]]["model"]()
        model._rb = True
        return model
    raise ValueError(f"unknown model {name}")


def load_weights(model: nn.Module, path: str, map_location="cpu"):
    """Accepts our checkpoints ({'model': sd}) and raw state_dicts from the RAMP / E-AT repos
    (keys without the leading '1.' of our Sequential wrapper)."""
    sd = torch.load(path, map_location=map_location)
    sd = sd.get("model", sd.get("state_dict", sd))
    if getattr(model, "_rb", False):  # RobustBench model fine-tuned by the official RAMP code: raw state_dict
        model.load_state_dict({k.replace("module.", "", 1): v for k, v in sd.items()})
        return model
    if not any(k.startswith("1.") for k in sd):
        sd = {f"1.{k}": v for k, v in sd.items()}
    own = model.state_dict()
    sd.update({k: v for k, v in own.items() if k.startswith("0.") and k not in sd})  # Normalize buffers
    model.load_state_dict(sd)
    return model


_BN_KEY = re.compile(r"^(.*\b(?:bn1|bn2|bn))\.(weight|bias|running_mean|running_var|num_batches_tracked)$")


def load_plain_into_gated(model: nn.Module, sd: dict):
    """Initialise a gated ("squint") model from a plain PreActResNet-18 state dict (the shared l_inf pretrain).
    Convs and the linear layer map one to one. Every plain BatchNorm becomes a GatedBN: running statistics go to its
    shared BatchNorm, and weight / bias are copied into every one of the K per-threat affine rows, so at initialisation the
    gated model computes exactly the plain model's function whatever the gate says. Only the gate head stays random.
    Raises if anything other than the gate head is left uninitialised."""
    new = {}
    for k, v in sd.items():
        m = _BN_KEY.match(k)
        if not m:
            new[k] = v
            continue
        base, leaf = m.groups()
        if leaf in ("weight", "bias"):
            K = model.state_dict()[base + (".gamma" if leaf == "weight" else ".beta")].shape[0]
            new[base + (".gamma" if leaf == "weight" else ".beta")] = v.unsqueeze(0).repeat(K, 1)
        else:
            new[f"{base}.bn.{leaf}"] = v
    missing, unexpected = model.load_state_dict(new, strict=False)
    bad = [k for k in missing if ".gate_" not in k] + list(unexpected)
    if bad:
        raise RuntimeError(f"gated init from plain checkpoint left keys unmatched: {bad[:8]}")
    return model
