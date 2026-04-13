import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

def get_cifar10_dataloaders(batch_size=64, test_batch_size=64, data_root="./data", num_workers=0, pin_memory=False, augmentation=False, train_subset_size=None, test_subset_size=None, seed=0):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]) if augmentation else transforms.Compose([transforms.ToTensor(), normalize])
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_transform)

    g = torch.Generator().manual_seed(seed)

    if train_subset_size is not None and train_subset_size < len(train_dataset):
        idx = torch.randperm(len(train_dataset), generator=g)[:train_subset_size].tolist()
        train_dataset = Subset(train_dataset, idx)

    if test_subset_size is not None and test_subset_size < len(test_dataset):
        idx = torch.randperm(len(test_dataset), generator=g)[:test_subset_size].tolist()
        test_dataset = Subset(test_dataset, idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers > 0))
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers > 0))
    return train_loader, test_loader