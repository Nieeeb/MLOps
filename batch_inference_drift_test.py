import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import argparse
import numpy as np
from nets.net import Net
from utils.util import load_params, visualize_batch
from utils.data import prepare_cifar10_loaders
from utils.model_tools import load_model
from typing import List
import torch.nn as nn

from typing import Optional, Tuple
from torch import Tensor


class DriftDetector:  # class, so that it can keep the state of accuracies across batches
    """
    Keeps a running mean & variance of batch accuracies (Welford’s algorithm)
    and flags drift when accuracy is more than `threshold` std dev below the mean.
    """

    def __init__(self, threshold: float = 2.0):
        self.threshold = threshold
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0  # sum of squares of differences from the current mean

    def __call__(self, outputs: Tensor, labels: Tensor) -> bool:
        preds = outputs.argmax(dim=1)
        acc = (preds == labels).float().mean().item()

        self.count += 1
        delta = acc - self.mean
        self.mean += delta / self.count
        delta2 = acc - self.mean
        self.M2 += delta * delta2

        if self.count < 2:
            return False
        variance = self.M2 / (self.count - 1)
        std_dev = variance**0.5

        # print(f"Accuracy is: {acc}, drift is detected if acc under: {(self.mean - self.threshold * std_dev)}")

        return acc < (self.mean - self.threshold * std_dev)


def batch_inference(
    inputs: Tensor,
    labels: Tensor,
    model: nn.Module,
    detector: DriftDetector,
    drift_test: bool = False,
) -> Tuple[Tensor, Optional[bool]]:
    """
    Run a model forward pass and optionally perform drift detection.

    Args:
        inputs: A batch of input tensors.
        labels: Corresponding ground-truth labels.
        criterion: Loss function used by the drift detector.
        drift_test: If True, run drift detection on outputs.

    Returns:
        A tuple of (model outputs, drift flag or None).
    """

    with torch.no_grad():
        outputs = model(inputs)

    flag: Optional[bool] = None
    if drift_test:
        flag = detector(outputs, labels)

    return outputs, flag


def main(drift_test: bool = False) -> None:
    """
    Load parameters, prepare data, and run batch inference
    (with optional drift testing) over the CIFAR-10 test set.
    """

    model = load_model(train=False, checkpoint="Weights\model.pt")

    params = load_params()
    criterion = nn.CrossEntropyLoss()

    drift_index = 8

    _, _, test_loader, _, _, _ = prepare_cifar10_loaders(
        batch_size=params["test_batch_size"],
        data_path=params["data_path"],
        num_workers=params["num_workers"],
        shuffle=False,
        drift_test=True,
        drift_index=drift_index,
    )

    detector = DriftDetector(threshold=params.get("drift_threshold", 2.0))

    for batch_idx, (inputs, labels) in enumerate(test_loader):
        outputs, drift_flag = batch_inference(
            inputs,
            labels,
            model=model,
            detector=detector,
            drift_test=drift_test,
        )
        if drift_test:
            if drift_flag:
                print(
                    f"Drift was detected at batch: {batch_idx}. Actual drift index: {drift_index}"
                )
                visualize_batch(inputs, outputs)
                break

            elif batch_idx == 8:
                visualize_batch(inputs, outputs)
                break


if __name__ == "__main__":
    main(drift_test=True)


"""
def load_model(checkpoint_path, device):
    model = Net().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def prepare_test_loader(batch_size=64, data_path="./data"):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    test_dataset = datasets.CIFAR10(
        root=data_path, train=False, download=True, transform=transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    return test_loader


def run_inference(model, test_loader, device):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(args.checkpoint, device)

    # Prepare test loader
    test_loader = prepare_test_loader(
        batch_size=args.batch_size, data_path=args.data_path
    )

    # Run inference
    preds, labels = run_inference(model, test_loader, device)

    # Evaluation
    acc = accuracy_score(labels, preds)
    print(f"Test Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", classification_report(labels, preds))
    print("\nConfusion Matrix:\n", confusion_matrix(labels, preds))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for inference"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data",
        help="CIFAR-10 data directory",
    )

    args = parser.parse_args()
    main(args)
"""
