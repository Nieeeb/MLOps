from pathlib import Path

from utils.data import prepare_cifar10_loaders
from utils.model_tools import load_model
from utils.util import load_params


def test() -> None:
    """Evaluate the trained model on the test set 
    and print accuracy/error rate."""
    checkpoint_path = Path("Weights/run_1/checkpoint_000.pt")

    net = load_model(
        train=False,
        checkpoint=checkpoint_path,
    )

    params = load_params()

    train_loader, valid_loader, test_loader = prepare_cifar10_loaders(
    batch_size=params.get("batch_size"),
    data_path=params.get("data_path"),
    num_workers=params.get("num_workers"),
    shuffle=params.get("shuffle"),
    )

    total_correct = 0
    total_samples = 0

    for inputs, labels in test_loader:
        outputs = net(inputs)

        # Accuracy for this batch
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    error_rate = 1.0 - accuracy

    print(f"Accuracy   : {accuracy:.4%}")
    print(f"Error rate : {error_rate:.4%}")
    
    
if __name__ == "__main__":
    test()