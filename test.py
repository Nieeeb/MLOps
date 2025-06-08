from pathlib import Path

from utils.data import prepare_cifar10_loaders
from utils.model_tools import load_model
from utils.util import load_params
import torch
import time


def test(net, test_loader) -> None:
    total_correct = 0
    total_samples = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clock_start = False
    for inputs, labels in test_loader:
        if clock_start == False:
            start_time = time.time()
            clock_start = True
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net(inputs)

        # Accuracy for this batch
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
    end_time = time.time()
    test_duration = end_time - start_time
    batch_size = len(test_loader.dataset) / len(test_loader)
    per_sample_time = test_duration / len(test_loader.dataset)
    per_batch_time = test_duration / len(test_loader)
    accuracy = total_correct / total_samples
    error_rate = 1.0 - accuracy

    print(f"Accuracy   : {accuracy:.4%}")
    print(f"Error rate : {error_rate:.4%}")
    print(f"Total test time: {test_duration}")
    print(f"Per batch time (N = {batch_size}): {per_batch_time}")
    print(
        f"Per sample time ({len(test_loader.dataset)} samples): {per_sample_time}"
    )


if __name__ == "__main__":
    """Evaluate the trained model on the test set
    and print accuracy/error rate."""
    checkpoint_path = Path("Weights/run_local_mini/checkpoint_last.pt")

    net = load_model(
        train=False,
        checkpoint=checkpoint_path,
    )

    params = load_params()

    train_loader, valid_loader, test_loader, _, _, _ = prepare_cifar10_loaders(
        batch_size=params.get("batch_size"),
        data_path=params.get("data_path"),
        num_workers=params.get("num_workers"),
        shuffle=params.get("shuffle"),
    )

    test(net, test_loader)
