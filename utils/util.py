import torch
import random
import numpy as np
import yaml
import matplotlib.pyplot as plt
import torchvision
from typing import Dict, Any
import math
from typing import Iterable, Optional, Tuple
from torch import Tensor


def load_params() -> Dict[str, Any]:
    with open("configs/params.yaml") as config:
        try:
            params = yaml.safe_load(config)
        except yaml.YAMLError as exc:
            print(exc)
    return params


def setup_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def visualize_imgs_from_loader(train_loader, batch_size):
    def imshow(img):
        img = img * 0.5 + 0.5  # inverse of (x-0.5)/0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    # get some random training images
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(" ".join(f"{classes[labels[j]]:5s}" for j in range(batch_size)))


def visualize_batch(
    images: Tensor,
    logits: Tensor,
    classes: Optional[Iterable[str]] = None,
    nrow: int = 8,
    img_norm_mean: float = 0.5,
    img_norm_std: float = 0.5,
) -> None:
    """
    Show a grid of images, then print their predicted class names.

    Args:
        images:  Tensor of shape (B, C, H, W).
        logits:  Tensor of shape (B, num_classes).
        classes: Optional list of class names.
        nrow:    Number of images per row in the grid.
    """
    batch_size = images.size(0)

    if classes is None:
        classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )

    # Un-normalize and make a single grid tensor
    unnorm = images * img_norm_std + img_norm_mean
    grid = torchvision.utils.make_grid(unnorm, nrow=nrow)

    # Show grid
    plt.figure(figsize=(nrow * 1.5, (batch_size // nrow + 1) * 1.5))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis("off")
    plt.show()

    # Compute preds and print them
    preds = logits.argmax(dim=1).cpu().tolist()
    names = [classes[p] for p in preds]
    print("Predictions:")
    # Print in rows of nrow
    for i in range(0, batch_size, nrow):
        row_names = names[i: i + nrow]
        print(" | ".join(f"{name:5s}" for name in row_names))


class AverageMeter:
    def __init__(self):
        self.num = 0
        self.sum = 0
        self.avg = 0

    def update(self, v, n):
        if not math.isnan(float(v)):
            self.num = self.num + n
            self.sum = self.sum + v * n
            self.avg = self.sum / self.num


# BLACK TEST HAHAHAHAHHAHAHAAHAHAHAHAHAHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
