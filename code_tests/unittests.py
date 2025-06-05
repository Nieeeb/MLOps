import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from nets.net import Net
from train import train  


def _dummy_loaders():
    """
    Return two DataLoaders (train, valid),
    each yielding exactly one small batch of random data.
    """
    # CIFAR-like shape: 4 images of size 3×32×32, labels in [0..9]
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    ds = torch.utils.data.TensorDataset(x, y)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
    return loader, loader


class TrainFunctionTests(unittest.TestCase):
    def test_train_checkpointing(self):
        net = Net()
        train_loader, valid_loader = _dummy_loaders()                 # Dummy data loaders
        params = {"epochs": 1, "batch_size": 4, "foo": "bar"}         # Example params

        with tempfile.TemporaryDirectory() as tmpdirname:            
            run_dir = Path(tmpdirname) / "my_run_dir"
            train(
                net=net,
                epochs=1,
                train_loader=train_loader,
                valid_loader=valid_loader,
                params=params,
                run_dir=str(run_dir),
                test_mode=True,
            )

            # Assert: checkpoint files exist
            ckpt0 = run_dir / "checkpoint_000.pt"
            ckpt_last = run_dir / "checkpoint_last.pt"
            self.assertTrue(ckpt0.is_file(), msg="checkpoint_000.pt missing")
            self.assertTrue(ckpt_last.is_file(), msg="checkpoint_last.pt missing")



