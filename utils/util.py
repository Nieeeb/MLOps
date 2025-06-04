import torch
import random
import numpy as np
import yaml
import matplotlib.pyplot as plt
import torchvision



def load_params():
    with open(r"configs\params.yaml") as config:
        try:
            params = yaml.safe_load(config)
        except yaml.YAMLError as exc:
            print(exc)
    return params



def setup_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def visualize_imgs(train_loader, batch_size):
    def imshow(img):
        img = img * 0.5 + 0.5          # inverse of (x-0.5)/0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # get some random training images
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))




