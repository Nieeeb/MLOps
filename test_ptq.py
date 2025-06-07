import torch
import tensorrt
from pathlib import Path
from utils.data import prepare_cifar10_loaders
from utils.model_tools import load_model
from utils.util import load_params


def test() -> None:
    """Evaluate the quantized model on the test set and print accuracy/error rate."""
    checkpoint_path = Path("Weights/run_1/checkpoint_000.pt")

    # Load and prepare the model
    net = load_model(train=False, checkpoint=checkpoint_path)
    net.eval()  # Explicitly set evaluation mode
    net.to("cuda")  # Move to GPU for calibration and inference

    # Load data
    params = load_params()
    _, _, test_loader, _, _, _ = prepare_cifar10_loaders(
        batch_size=params.get("batch_size"),
        data_path=params.get("data_path"),
        num_workers=params.get("num_workers"),
        shuffle=params.get("shuffle"),
    )

    # Define calibration dataset (use a subset of test_loader)
    class CalibrationDataset:
        def __init__(self, dataloader, max_samples=1000):
            self.dataloader = dataloader
            self.max_samples | max_samples
            self.current = 0
            self.iterator = iter(dataloader)

        def __iter__(self):
            return self

        def __next__(self):
            if self.current >= self.max_samples:
                raise StopIteration
            batch, _ = next(self.iterator)
            self.current += batch.size(0)
            return [batch.to("cuda")]  # TensorRT expects a list of tensors

    calibration_dataset = CalibrationDataset(test_loader, max_samples=1000)

    # Compile model with Torch-TensorRT for INT8
    trt_module = tensorrt.compile(
        net,
        inputs=[
            tensorrt.Input(
                min_shape=[1, 3, 32, 32],
                opt_shape=[params.get("batch_size", 32), 3, 32, 32],
                max_shape=[params.get("batch_size", 32), 3, 32, 32],
                dtype=torch.float32,
            )
        ],
        enabled_precisions={torch.int8},  # Enable INT8 quantization
        calibrator=tensorrt.ptq.DataLoaderCalibrator(
            calibration_dataset,
            cache_file="calibration.cache",
            algo_type=tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
        ),
    )

    # Evaluate quantized model
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to("cuda"), labels.to("cuda")
            outputs = trt_module(inputs)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    error_rate = 1.0 - accuracy

    print(f"Quantized Model Accuracy: {accuracy:.4%}")
    print(f"Quantized Model Error Rate: {error_rate:.4%}")


if __name__ == "__main__":
    test()
