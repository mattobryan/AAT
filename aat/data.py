"""CIFAR loaders. A fixed slice of the *training* set is held out as the feedback set that
drives the adaptive schedulers, so the test set is never seen during training."""
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T


def _dataset_cls(name):
    return {"cifar10": torchvision.datasets.CIFAR10, "cifar100": torchvision.datasets.CIFAR100}[name]


def get_loaders(cfg):
    d = cfg["data"]
    root, name = d.get("root", "./data"), d.get("dataset", "cifar10")
    n_feedback = d.get("feedback_size", 1000)
    workers = d.get("num_workers", 4)
    cls = _dataset_cls(name)

    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()])
    test_tf = T.ToTensor()
    train_aug = cls(root, train=True, download=True, transform=train_tf)
    train_plain = cls(root, train=True, download=True, transform=test_tf)
    test = cls(root, train=False, download=True, transform=test_tf)

    g = torch.Generator().manual_seed(d.get("split_seed", 0))
    perm = torch.randperm(len(train_aug), generator=g).tolist()
    fb_idx, tr_idx = perm[:n_feedback], perm[n_feedback:]

    bs = cfg["train"]["batch_size"]
    train_loader = DataLoader(Subset(train_aug, tr_idx), bs, shuffle=True, num_workers=workers,
                              pin_memory=True, drop_last=True, persistent_workers=workers > 0)
    feedback_loader = DataLoader(Subset(train_plain, fb_idx), 500, shuffle=False, num_workers=workers)
    test_loader = DataLoader(test, 500, shuffle=False, num_workers=workers)
    return train_loader, feedback_loader, test_loader


def get_test_tensors(cfg, n: int):
    """First n test points as tensors (the RobustBench / AutoAttack convention)."""
    d = cfg["data"]
    test = _dataset_cls(d.get("dataset", "cifar10"))(d.get("root", "./data"), train=False,
                                                     download=True, transform=T.ToTensor())
    loader = DataLoader(Subset(test, range(n)), n, shuffle=False)
    return next(iter(loader))
