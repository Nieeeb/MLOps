from torchvision.datasets import CIFAR10
from utils.util import setup_seed
import os
from pathlib import Path
from torch.utils import data
import torch
import torchvision.transforms as transforms
import torch.distributed as dist
from PIL import Image
from torch.utils.data import Subset


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


class CIFAR10Drift(CIFAR10):
    def __init__(
        self,
        root,
        train: bool,
        download: bool,
        default_transform,
        drift_transform,
        drift_index: int,  # now means “batch-number” to start drift
        drift_test: bool,
        batch_size: int,  # NEW: how many samples per batch
    ):
        super().__init__(
            root=root, train=train, transform=None, download=download
        )
        self.default_transform = default_transform
        self.drift_transform = drift_transform
        self.drift_index = drift_index
        self.drift_test = drift_test
        self.batch_size = batch_size  # store for thresholding

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.drift_test and index >= (self.drift_index * self.batch_size):
            t = self.drift_transform
        else:
            t = self.default_transform

        return t(img), target


def prepare_cifar10_loaders(
    batch_size: int = 50,
    num_workers: int = 2,
    data_path: str = "Data/CIFAR10",
    shuffle: bool = True,
    drift_test: bool = False,
    drift_index: int = -1,
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

    default_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    drift_transform = transforms.Compose(
        [
            # Strong random crop & resize
            transforms.RandomResizedCrop(
                size=32, scale=(0.5, 1.0), ratio=(0.75, 1.333)
            ),
            # Random flips & rotations
            transforms.RandomHorizontalFlip(p=0.7),
            transforms.RandomRotation(degrees=30),
            # Heavy color jitter
            transforms.ColorJitter(
                brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2
            ),
            # Occasionally turn to grayscale
            transforms.RandomGrayscale(p=0.3),
            # **NEW: brutal color hacks**
            transforms.RandomSolarize(threshold=128, p=0.5),
            transforms.RandomPosterize(bits=2, p=0.5),
            # transforms.RandomInvert(p=0.3),    # you can swap in this too
            # Add a bit of blur
            transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0)),
            # To tensor & normalize
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # Randomly erase a patch
            transforms.RandomErasing(
                p=0.5,
                scale=(0.02, 0.25),
                ratio=(0.3, 3.3),
            ),
        ]
    )

    full_train = CIFAR10(
        root=train_path,
        train=True,
        download=False,
        transform=default_transform,
    )

    if drift_test:  # use our drift‐dataset
        test_data = CIFAR10Drift(
            root=train_path,
            train=False,
            download=False,
            default_transform=default_transform,
            drift_transform=drift_transform,
            drift_index=drift_index,
            drift_test=drift_test,
            batch_size=batch_size,
        )
    else:  # normal dataset
        test_data = CIFAR10(
            root=test_path,
            train=False,
            download=False,
            transform=default_transform,
        )

    if drift_test:
        train_indices = list(range(0, 40000))
        valid_indices = list(range(40000, 50000))

        train_data = Subset(full_train, train_indices)
        valid_data = Subset(full_train, valid_indices)
    else:
        generator = torch.Generator().manual_seed(0)
        train_data, valid_data = data.random_split(
            full_train, lengths=[40000, 10000], generator=generator
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
