import os
import time
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from carbontracker.tracker import CarbonTracker

from utils.data import prepare_cifar10_loaders
from utils.model_tools import load_model
from utils.util import load_params
from utils.wandb_logging import wandb_init, wandb_log
from utils.minio_tools import create_session, upload_local_directory_to_minio


def train(
    net: nn.Module,
    epochs: int,
    train_loader,
    valid_loader,
    params: dict[str, Any],
    run_dir: str | None,
    test_mode: bool = False,
) -> None:
    """Train *net* for *epochs* and save checkpoints in *run_dir*.

    Args:
        net: The model to train.
        epochs: Number of full passes over the training data.
        train_loader: DataLoader yielding training batches.
        valid_loader: DataLoader yielding validation batches.
        params: Hyper-parameters to persist alongside the run.
        run_dir: Destination directory for logs and checkpoints.
                 If *None*, a timestamped folder under ``runs/`` is created.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    tracker = CarbonTracker(epochs=epochs)

    if run_dir is None:
        stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = os.path.join("runs", stamp)
    os.makedirs(run_dir, exist_ok=True)

    # Save the run parameters for reproducibility
    params_path = os.path.join(run_dir, "params.yaml")
    with open(params_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(params, f)

    for epoch in range(epochs):
        # ── training ───────────────────────────────────────────────
        tracker.epoch_start()
        net.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if test_mode:
                break

        train_loss = running_loss / len(train_loader)

        # ── validation ─────────────────────────────────────────────
        net.eval()
        vloss, nsamp = 0.0, 0

        with torch.no_grad():
            for inputs, labels in valid_loader:
                outputs = net(inputs)
                batch_size = inputs.size(0)
                vloss += criterion(outputs, labels).item() * batch_size
                nsamp += batch_size

                if test_mode:
                    break

        val_loss = vloss / nsamp

        tracker.epoch_end()

        wandb_log(val_loss=val_loss, epoch=epoch, loss_avg=train_loss)

        # ── checkpoint ────────────────────────────────────────────
        ckpt = {
            "epoch": epoch,
            "model": net.state_dict(),
            "optim": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        torch.save(ckpt, os.path.join(run_dir, f"checkpoint_{epoch:03d}.pt"))
        torch.save(ckpt, os.path.join(run_dir, "checkpoint_last.pt"))

    client, credentials = create_session()
    upload_local_directory_to_minio(
        local_path=run_dir,
        credentials=credentials,
        minio_path="Weights/",
        client=client,
    )
    tracker.stop()
    print("Training finished.")


def main():
    params = load_params()

    wandb_init(params)

    net = load_model(train=True)

    train_loader, valid_loader, test_loader, _, _, _ = prepare_cifar10_loaders(
        batch_size=params.get("batch_size"),
        data_path=params.get("data_path"),
        num_workers=params.get("num_workers"),
        shuffle=params.get("shuffle"),
    )

    train(
        net,
        params.get("epochs"),
        train_loader,
        valid_loader,
        params=params,
        run_dir=params.get("run_dir"),
        test_mode=False,
    )


if __name__ == "__main__":
    main()
