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

def draw_boxes(image, bbox):
    img_copy = image.copy()
    for i in range(len(bbox)):
        x, y, w, h, c, b = bbox[i].cpu().detach().numpy().astype('int')
        print(bbox[i].cpu().detach().numpy().astype('int'))
        cv2.rectangle(img_copy, (x, y), (w, h), (0, 0, 255), 5)
    return img_copy


@hydra.main(config_path="../../config", config_name="config")
def main(args):
    model = yolo_v8_n(num_classes=4).cuda()

    #train_dataset, val_dataset, test_dataset = load_datasets(args)
    model.eval()
    path = '/home/nieb/Projects/Big Data/Images/Seasons_drift/v2/harborfrontv2/frames/20200514/clip_0_1331/image_0110.jpg'
    image = PIL.Image.open(path)
    image.show()

    image_gray = F.to_grayscale(image, num_output_channels=3)

    tensor = F.to_tensor(image_gray).cuda()
    tensor = tensor.unsqueeze(0)
    print(tensor.shape)

    prediction = model(tensor)
    prediction = non_max_suppression(prediction, 0.001, 0.65)
    #for idx in range(prediction[0].shape[1]):
    #    print(prediction[0][idx])

    
    boxes_image = draw_boxes(image, prediction[0])
    boxes_image.show()
    


if __name__ == "__main__":
    main()