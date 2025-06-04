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
