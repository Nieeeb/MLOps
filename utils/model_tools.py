from nets.net import Net
import torch
from pathlib import Path
import torch.nn as nn


def load_model(
    train: bool = True,
    checkpoint: str | Path | None = None,
) -> nn.Module:
    """Return a Net model on the available device.

    If `checkpoint` is provided, load model weights from that file.
    The model is set to training mode by default unless `train=False`.
    """
    if checkpoint:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # 1. read the file → dict
        ckpt = torch.load(checkpoint_path, map_location=device)

        # 2. rebuild architecture and fill in weights
        model = Net().to(device)
        model.load_state_dict(ckpt["model"])  # <─ no second torch.load!

    else:
        model = Net()

    # 3. set mode
    model.train(mode=train)
    return model
