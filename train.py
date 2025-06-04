import torch.utils
import torch.utils.data
from nets.net import Net
import torch
import yaml
from utils.util import load_params
from utils.model_tools import load_model
from typing import Any
import torchvision
from torch.utils.data import DataLoader
from utils.data import prepare_cifar10_loaders
import torch.optim as optim
import torch.nn as nn
import os



def load_data(params: dict[str, Any]):
    train_loader, valid_loader, test_loader = prepare_cifar10_loaders(
    batch_size=params.get("batch_size"),
    data_path=params.get("data_path"),
    num_workers=params.get("num_workers"),
    shuffle=params.get("shuffle"),
    )

    return train_loader, valid_loader, test_loader



def train(
        net: nn.Module, 
        epochs: int, 
        train_loader, 
        valid_loader,
        params: dict[str, Any],
        run_dir: str):
    


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


    if run_dir is None:
        stamp  = time.strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = os.path.join("runs", stamp)
    os.makedirs(run_dir, exist_ok=True)



    with open(os.path.join(run_dir, "params.yaml"), "w") as f:
            yaml.safe_dump(params, f)


    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(net(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            break


        train_loss = running_loss / len(train_loader)

        # ── validation ──────────────────────────────────────────────
        net.eval()
        vloss, nsamp = 0.0, 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                batch = inputs.size(0)
                vloss += criterion(net(inputs), labels).item() * batch
                nsamp += batch

                break




        val_loss = vloss / nsamp

        print(f"Epoch {epoch:3d} │ train {train_loss:.4f} │ val {val_loss:.4f}")

        # ── write checkpoint ────────────────────────────────────────
        ckpt = {
            "epoch" : epoch,
            "model" : net.state_dict(),
            "optim" : optimizer.state_dict(),
            "train_loss" : train_loss,
            "val_loss"   : val_loss,
        }
        torch.save(ckpt, os.path.join(run_dir, f"checkpoint_{epoch:03d}.pt"))
        torch.save(ckpt, os.path.join(run_dir, "checkpoint_last.pt"))   
        
        break




def main():

    params = load_params()
    net = load_model(train=True)
    train_loader, valid_loader, test_loader = load_data(params)

    train(net, 
          params.get('epochs'),
          train_loader, 
          valid_loader,
          params=params,
          run_dir=params.get('run_dir'))









if __name__ == "__main__":
    main()




# Vi skal også huske at gemme trænings konfigurationen
