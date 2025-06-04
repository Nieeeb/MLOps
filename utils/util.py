from nets.net import Net
import torch



def load_model(train: bool=True,
               checkpoint: 
               ):
    
    model = Net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if train:
        model.train()
    else:
        model.eval()
    return model
import random
import numpy as np
import torch


def setup_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True