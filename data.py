from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar10_dataloaders(batch_size=256, test_batch_size=256, data_root="./data", num_workers=2, pin_memory=False):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers > 0))
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=(num_workers > 0))
    return train_loader, test_loader