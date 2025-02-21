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
from tqdm import tqdm
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

from loss import YoloCriterion

from Dataloading.coco import COCODataset
import matplotlib.pyplot as plt

from matplotlib import patches
import numpy as np

from model.YoloV8.util import ComputeLoss


'''
To-DO
- Formatter outputs så der kan beregnes loss
https://github.com/MarcoParola/conditioning-transformer.git
'''

def visualize(inputs: torch.Tensor, outputs: torch.Tensor, class_names: list):


    inputs = inputs.squeeze(0).permute(1, 2, 0)
    image = inputs.numpy()


    outputs = outputs.permute(2, 0, 1)
    pred = outputs[1267].squeeze()



    pred_np = pred.detach().cpu().numpy()
    cx, cy, w, h = pred_np[:4]
    scores = pred_np[4:]
    
    # Determine the best scoring class
    class_idx = np.argmax(scores)
    score = scores[class_idx]
    
    # Convert the bounding box from center format (c_x, c_y, w, h)
    # to top-left format (x, y, w, h) for plotting:
    x = cx - w / 2
    y = cy - h / 2
    
    # Create a plot
    fig, ax = plt.subplots(1, figsize=(8, 8))
    # If your image is in [0,1] range, you may want to multiply by 255:
    if image.max() <= 1.0:
        image_disp = (image * 255).astype(np.uint8)
    else:
        image_disp = image.astype(np.uint8)
    ax.imshow(image_disp)
    
    # Create and add the rectangle patch for the bounding box
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    
    # Prepare a label text with the class name and confidence score
    label = f"{class_names[class_idx]}: {score:.2f}"
    ax.text(x, y - 5, label, color='red', fontsize=12, backgroundcolor='white')
    
    ax.axis('off')
    plt.show()

#@hydra.main(config_path="../../config", config_name="config")
def main():
    class_names = ['person',  'bicycle', 'motorcycle', 'vehicle']
    # Creating a model and defining number of classes
    model = yolo_v8_n(num_classes=4).cuda()

    #train_dataset, val_dataset, test_dataset = load_datasets(args)
    model.eval()


    object = 'Test.json'
    object_path = 'harborfrontv2/' + object
    output_path = 'outputs/' + object
    stupid_patth = '/home/nieb/Projects/DAKI Mini Projects/MLOps/outputs/Valid.json'
    train_dataset = COCODataset(root='', annotation=stupid_patth, numClass=4, online=True)
    train_loader = data.DataLoader(train_dataset, batch_size=1)




    epochs = 1
    with open(os.path.abspath('/home/nieb/Projects/DAKI Mini Projects/MLOps/model/YoloV8/args.yaml')) as f:
        params = yaml.safe_load(f)

    criterion = YoloCriterion(params, model)
    # criterion = ComputeLoss(model, params) # Loss fra yolo github virker, men giver infinite loss. Umiddelbart fordi targets er forkert format

    for epoch in range(epochs):
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            model.train()
            inputs = inputs.cuda()
            #targets = targets.cuda()
            outputs = model(inputs)
            #print(targets['boxes'])
            #print(targets['labels'])
            #print(outputs[0].shape)
            #cattarget = torch.cat((targets['boxes'], targets['labels']), 1)
            loss = criterion(outputs, targets)
            print(loss.items())
            break
            #print(loss)
            loss.backward()
            if batch_idx == 200:
                visualize(inputs, outputs, class_names)
                break
    # path = '/home/nieb/Projects/Big Data/Images/Seasons_drift/v2/harborfrontv2/frames/20200514/clip_0_1331/image_0110.jpg'
    # image = PIL.Image.open(path)
    # image.show()
    # Image is 1 color channel, here we convert to 3 color channels. Should probably change model input dim instead

    # tensor = F.to_tensor(image_gray).cuda()
    # # Add batch dimension
    # tensor = tensor.unsqueeze(0)
    # print(tensor.shape)

    # Forward pass through the model
    # prediction = model(tensor)
    # Built in function to remove some of the overlapping boxes
    # prediction = non_max_suppression(prediction, 0.001, 0.65)
    #for idx in range(prediction[0].shape[1]):
    #    print(prediction[0][idx])
    
    # Right now, current problems are:
    # Dont know what the output consists off. What is in each tensor dimension?
    # Do we need to scale output boxes to fit the image?

    


if __name__ == "__main__":
    main()