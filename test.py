import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms

from nets.net import Net

classes = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


def load_model(train: bool = False, checkpoint:
                str | Path | None = None) -> nn.Module:
    """Load a neural network model, optionally from a checkpoint.

    Args:
        train (bool): If True, set model to training mode; otherwise, 
        evaluation mode.
        checkpoint (str | Path | None): Path to a saved model checkpoint. 
        If None, initialize a new model.

    Returns:
        nn.Module: The loaded or initialized model, 
        moved to the appropriate device.
    """
    model = Net()
    if checkpoint:
        model = torch.load(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train() if train else model.eval()
    return model


def unpickle(file):
    """Load a pickled file with bytes encoding.

    Args:
        file (str or Path): Path to the pickled file.

    Returns:
        dict: The unpickled dictionary.
    """
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')


def load_test_batch(path):
    """Load the CIFAR-10 test batch from a specified path.

    Args:
        path (str or Path): Directory containing the test batch file.

    Returns:
        tuple: A tuple of (data, labels) where data is a NumPy array of images
               and labels is a list of corresponding labels.
    """
    batch = unpickle(Path(path) / 'test_batch')
    data = batch[b'data'].reshape(-1, 3, 32, 32).astype(
        np.uint8)
    labels = batch[b'labels']
    return data, labels


if __name__ == "__main__":
    checkpoint_path = Path("models/latest_model.pth")
    model = load_model(train=False, checkpoint=checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        )
    ])

    data, labels = load_test_batch("Data/CIFAR10/test/cifar-10-batches-py")
    images = torch.stack([
        transform(img) for img in data
    ])
    labels = torch.tensor(labels)

    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    overall_acc = accuracy_score(all_labels, all_preds)
    print(f"Overall accuracy: {overall_acc * 100:.2f}%")

    conf_mat = confusion_matrix(all_labels, all_preds)
    per_class_acc = conf_mat.diagonal() / conf_mat.sum(axis=1)

    print("Per-class accuracy:")
    for idx, acc in enumerate(per_class_acc):
        print(f"{classes[idx]:>5s}: {acc * 100:.2f}%")

    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=classes,
                yticklabels=classes, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()