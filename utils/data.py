from torchvision.datasets import CIFAR10
from utils.util import setup_seed
import os
from pathlib import Path
from torch.utils import data
import torch
import torchvision.transforms as transforms


def download_cifar10(path: str = "Data/CIFAR10"):
    if not Path(path).exists():
        parent = os.path.dirname(path)
        if not Path(parent).exists():
            os.mkdir(parent)
        os.mkdir(path)

    train_path = os.path.join(path, 'train')
    test_path = os.path.join(path, 'test')

    train = CIFAR10(root=train_path,
                    train=True,
                    download=True
                    )

    test = CIFAR10(root=test_path,
                   train=False,
                   download=True
                   )


def check_path(path: str) -> bool:
    if os.path.exists(path):
        return True
    else:
        return False


def prepare_cifar10_loaders(batch_size: int = 50, num_workers: int = 2, data_path: str = "Data/CIFAR10", shuffle: bool = True) -> tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
    setup_seed()
    # Check if data has been downloaded already
    if check_path(data_path) == False:
        download_cifar10(data_path)

    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data = CIFAR10(root=train_path,
                         train=True,
                         download=False,
                         transform=transform
                         )

    generator = torch.Generator().manual_seed(0)
    train_data, valid_data = data.random_split(
        train_data, lengths=[40000, 10000], generator=generator)

    test_data = CIFAR10(root=test_path,
                        train=False,
                        download=False,
                        transform=transform
                        )

    train_loader = data.DataLoader(train_data,
                                   batch_size=batch_size,
                                   shuffle=shuffle,
                                   num_workers=num_workers,
                                   )

    valid_loader = data.DataLoader(valid_data,
                                   batch_size=batch_size,
                                   shuffle=shuffle,
                                   num_workers=num_workers
                                   )

    test_loader = data.DataLoader(test_data,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers
                                  )

    return train_loader, valid_loader, test_loader


def main():
    setup_seed()
    train, valid, test = prepare_cifar10_loaders()
    print(len(train.dataset))


if __name__ == "__main__":
    main()
