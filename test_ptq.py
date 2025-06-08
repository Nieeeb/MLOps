import torch
from pathlib import Path
from utils.data import prepare_cifar10_loaders
from utils.model_tools import load_model
from utils.util import load_params
from test import test
from utils.ptq import mct_ptq, onnx_ptq_export


def test_ptq() -> None:
    """Evaluate the quantized model on the test set and print accuracy/error rate."""
    checkpoint_path = Path("Weights/run_local_mini/checkpoint_last.pt")

    # Load and prepare the model
    net = load_model(train=False, checkpoint=checkpoint_path)
    net.eval()  # Explicitly set evaluation mode
    net.to("cuda")  # Move to GPU for calibration and inference

    # Load data
    params = load_params()
    _, validation_loader, test_loader, _, _, _ = prepare_cifar10_loaders(
        batch_size=params.get("batch_size"),
        data_path=params.get("data_path"),
        num_workers=params.get("num_workers"),
        shuffle=params.get("shuffle"),
    )
    # full_model, quantized_model = ptq_model(model_to_quantize=net, calibration_loader=validation_loader)
    quantized_model, quant_info = mct_ptq(net, validation_loader)

    # Export the model as onnx
    onnx_ptq_export(
        in_model=net,
        calibration_loader=validation_loader,
        export_path="model.onnx",
        quant_model=False,
    )
    onnx_ptq_export(
        in_model=quantized_model,
        calibration_loader=validation_loader,
        export_path="model_quant.onnx",
        quant_model=False,
    )  # Model is already quantized

    print("--- Full Model ---")
    test(net, test_loader)

    # Note that the quantized model is slower when running with Pytorch. This is due to the layers added simply simulating
    # being quantized, and not being smaller. This adds some overhead. Models only become faster when exported and run
    # using a framework such as onnx. However, accuracy achieved is the same as when the model is exported
    print("--- Quant Model ---")
    test(quantized_model, test_loader)


if __name__ == "__main__":
    test_ptq()
