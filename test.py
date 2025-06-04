import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from utils.model_tools import load_model
from utils.util import load_params
from train import load_data

from nets.net import Net



def test():
    checkpoint_path = Path("Weights/run_1/checkpoint_000.pt")
    net = load_model(train=False,
                     checkpoint=checkpoint_path)
    params = load_params()
    train_loader, valid_loader, test_loader = load_data(params)



    total_correct = 0
    total_samples = 0
    
    for inputs, labels in valid_loader:
        outputs = net(inputs)
        # accuracy
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

        

        accuracy   = total_correct / total_samples
        error_rate = 1.0 - accuracy

        print(f"Accuracy        : {accuracy:.4%}")
        print(f"Error rate      : {error_rate:.4%}")
            

        break
    
if __name__ == "__main__":
    test()