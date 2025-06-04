from nets.net import Net
import torch
from pathlib import Path
import torch.nn as nn


def load_model(
        train: bool=True,
        checkpoint: str | Path | None = None, 
) -> nn.Module:
    """Return a Net model on the available device.

    If `checkpoint` is provided, load model weights from that file.
    The model is set to training mode by default unless `train=False`.
    """
    model = Net()

    if checkpoint:
        model = torch.load(checkpoint)
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if train:
        model.train()
    else:
        model.eval()

    return model