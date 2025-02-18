import sys
import os

import PIL.ImageOps
root_dir = os.getcwd()
sys.path.append(root_dir)
import argparse
import copy
import csv
import warnings

import PIL.Image
import numpy
import torch
import tqdm
import yaml
from torch.utils import data
from Dataloading.utils import load_datasets
import hydra
import PIL
import torchvision.transforms.functional as F
import torchvision.transforms as T
import cv2
from util import non_max_suppression

from nn import yolo_v8_n

@hydra.main(config_path="../../config", config_name="config")
def main(args):
    # Creating a model and defining number of classes
    model = yolo_v8_n(num_classes=4).cuda()

    #train_dataset, val_dataset, test_dataset = load_datasets(args)
    model.eval()
    path = '/home/nieb/Projects/Big Data/Images/Seasons_drift/v2/harborfrontv2/frames/20200514/clip_0_1331/image_0110.jpg'
    image = PIL.Image.open(path)
    image.show()
    # Image is 1 color channel, here we convert to 3 color channels. Should probably change model input dim instead
    image_gray = F.to_grayscale(image, num_output_channels=3)

    tensor = F.to_tensor(image_gray).cuda()
    # Add batch dimension
    tensor = tensor.unsqueeze(0)
    print(tensor.shape)

    # Forward pass through the model
    prediction = model(tensor)
    # Built in function to remove some of the overlapping boxes
    prediction = non_max_suppression(prediction, 0.001, 0.65)
    #for idx in range(prediction[0].shape[1]):
    #    print(prediction[0][idx])
    
    # Right now, current problems are:
    # Dont know what the output consists off. What is in each tensor dimension?
    # Do we need to scale output boxes to fit the image?

    


if __name__ == "__main__":
    main()