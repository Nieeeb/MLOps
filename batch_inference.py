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
from nets.net import Net  # Adjust if you're using another architecture


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
