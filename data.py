import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def build_cifar10_transforms(augmentation=False):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    if augmentation:
        train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    else:
        train_transform = transforms.Compose([transforms.ToTensor(), normalize])
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    return train_transform, test_transform


def subset_dataset(dataset, subset_size, seed):
    if subset_size is None or subset_size >= len(dataset):
        return dataset
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=g)[:subset_size].tolist()
    return Subset(dataset, idx)


def get_cifar10_dataloaders(batch_size=64, test_batch_size=64, data_root="./data", num_workers=0, pin_memory=False, augmentation=False, train_subset_size=None, test_subset_size=None, seed=0):
    train_transform, test_transform = build_cifar10_transforms(augmentation=augmentation)

    train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_transform)

    train_dataset = subset_dataset(train_dataset, train_subset_size, seed)
    test_dataset = subset_dataset(test_dataset, test_subset_size, seed)

    train_loader_generator = torch.Generator().manual_seed(seed + 1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers > 0), generator=train_loader_generator)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers > 0))

    return train_loader, test_loader
