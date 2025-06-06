from torchvision.datasets import CIFAR10
from utils.util import setup_seed
import os
from pathlib import Path
from torch.utils import data
import torch
import torchvision.transforms as transforms
import torch.distributed as dist


def download_cifar10(path: str = "Data/CIFAR10"):
    if not Path(path).exists():
        parent = os.path.dirname(path)
        if not Path(parent).exists():
            os.mkdir(parent)
        os.mkdir(path)

    train_path = os.path.join(path, "train")
    test_path = os.path.join(path, "test")

    train = CIFAR10(root=train_path, train=True, download=True)

    test = CIFAR10(root=test_path, train=False, download=True)


def check_path(path: str) -> bool:
    if os.path.exists(path):
        return True
    else:
        return False


def prepare_cifar10_loaders(
    batch_size: int = 50,
    num_workers: int = 2,
    data_path: str = "Data/CIFAR10",
    shuffle: bool = True,
) -> tuple[
    data.DataLoader,
    data.DataLoader,
    data.DataLoader,
    data.Sampler | None,
    data.Sampler | None,
    data.Sampler | None,
]:
    setup_seed()
    # Check if data has been downloaded already
    if check_path(data_path) == False:
        download_cifar10(data_path)

    train_path = os.path.join(data_path, "train")
    test_path = os.path.join(data_path, "test")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    full_train = CIFAR10(
        root=train_path, train=True, download=False, transform=transform
    )

    generator = torch.Generator().manual_seed(0)
    train_data, valid_data = data.random_split(
        full_train, lengths=[40000, 10000], generator=generator
    )

    test_data = CIFAR10(
        root=test_path, train=False, download=False, transform=transform
    )

    if (
        dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    ):
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # Create DistributedSampler for each subset
        train_sampler = data.distributed.DistributedSampler(
            train_data,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
        )
        valid_sampler = data.distributed.DistributedSampler(
            valid_data,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,  # usually validation isn’t shuffled
        )
        test_sampler = data.distributed.DistributedSampler(
            test_data,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,  # test shouldn’t be shuffled
        )

        # When using a sampler shuffle=False in DataLoader
        train_loader = data.DataLoader(
            train_data,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
        valid_loader = data.DataLoader(
            valid_data,
            batch_size=batch_size,
            sampler=valid_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
        test_loader = data.DataLoader(
            test_data,
            batch_size=batch_size,
            sampler=test_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )

    else:
        # Single‐GPU (or non‐distributed) fallback: no sampler
        train_sampler = None
        valid_sampler = None
        test_sampler = None

        train_loader = data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
        valid_loader = data.DataLoader(
            valid_data,
            batch_size=batch_size,
            shuffle=False,  # no shuffling of validation in single‐GPU
            num_workers=num_workers,
            pin_memory=True,
        )
        test_loader = data.DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return (
        train_loader,
        valid_loader,
        test_loader,
        train_sampler,
        valid_sampler,
        test_sampler,
    )


def main():
    setup_seed()
    train, valid, test = prepare_cifar10_loaders()
    print(len(train.dataset))


if __name__ == "__main__":
    main()
