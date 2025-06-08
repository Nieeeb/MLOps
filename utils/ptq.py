import torch
import model_compression_toolkit as mct


def mct_ptq(
    in_model: torch.nn.Module,
    calibration_loader: torch.utils.data.DataLoader,
    n_iter: int = 10,
):
    def reprensentative_data_gen():
        dataloader_iter = iter(calibration_loader)
        for _ in range(n_iter):
            yield [next(dataloader_iter)[0]]

    quant_model, quant_info = mct.ptq.pytorch_post_training_quantization(
        in_module=in_model, representative_data_gen=reprensentative_data_gen
    )

    return quant_model, quant_info


def onnx_ptq_export(
    in_model: torch.nn.Module,
    calibration_loader: torch.utils.data.DataLoader,
    n_iter: int = 10,
    export_path: str = "model_quant.onnx",
    quant_model: bool = False,
):
    def reprensentative_data_gen():
        dataloader_iter = iter(calibration_loader)
        for _ in range(n_iter):
            yield [next(dataloader_iter)[0]]

    if quant_model:
        quant_model, quant_info = mct.ptq.pytorch_post_training_quantization(
            in_module=in_model,
            representative_data_gen=reprensentative_data_gen,
        )

        mct.exporter.pytorch_export_model(
            quant_model,
            save_model_path=export_path,
            repr_dataset=reprensentative_data_gen,
        )
    else:
        mct.exporter.pytorch_export_model(
            in_model,
            save_model_path=export_path,
            repr_dataset=reprensentative_data_gen,
        )
