import unittest
import tempfile
import torch
import torch.nn as nn
import yaml
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))


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
        from nets.net import Net
        from train import train

        net = Net()
        train_loader, valid_loader = _dummy_loaders()  # Dummy data loaders
        params = {"epochs": 1, "batch_size": 4, "foo": "bar"}  # Example params

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
            self.assertTrue(
                ckpt_last.is_file(), msg="checkpoint_last.pt missing"
            )

            # Load the checkpoint and verify keys
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            ckpt_dict = torch.load(ckpt0, map_location=device)

            for key in ("epoch", "model", "optim", "train_loss", "val_loss"):
                self.assertIn(
                    key, ckpt_dict, msg=f"Key {key} not found in checkpoint"
                )

            net.load_state_dict(ckpt_dict["model"])

            net.to(device)
            net.eval()
            dummy_input = torch.randn(2, 3, 32, 32, device=device)

            with torch.no_grad():
                output = net(dummy_input)

            self.assertEqual(
                tuple(output.shape),
                (2, 10),
                msg=f"Expected output shape (2,10), got {tuple(output.shape)}",
            )


if __name__ == "__main__":
    unittest.main()
