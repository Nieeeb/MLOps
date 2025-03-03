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
from PIL import Image, ImageDraw

from Dataloading.coco import COCODataset, collateFunction
import matplotlib.pyplot as plt

from matplotlib import patches
import numpy as np

from model.YoloV8.util import ComputeLoss
from torch.optim import Adam

from ultraloss import v8DetectionLoss


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



    pred_np = pred.detach().cuda().numpy()
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

def print_tensor_as_float(tensor, precision=6):
    # Detach tensor and convert to numpy
    np_arr = tensor.detach().cuda().numpy()
    # Create a formatter that prints floats with the desired precision
    formatter = {'float_kind': lambda x: format(x, f'.{precision}f')}
    formatted = np.array2string(np_arr, formatter=formatter)
    print(formatted)


def targets_to_tensor(targets):
    """
    Converts a tuple of dictionaries (each with keys 'boxes' and 'labels') into a single tensor.
    Each output row is: [batch_index, x1, y1, x2, y2, label]
    
    Args:
        targets (tuple): Tuple of dictionaries. Each dictionary must contain:
                         - 'boxes': tensor of shape [num_boxes, 4]
                         - 'labels': tensor of shape [num_boxes]
                         
    Returns:
        Tensor: A tensor of shape [total_num_boxes, 6] containing the batch index, box coordinates, and label.
    """
    target_list = []
    for batch_idx, target in enumerate(targets):
        boxes = target['boxes']  # shape: [num_boxes, 4]
        labels = target['labels']  # shape: [num_boxes]
        # Create a tensor filled with the batch index for each box
        batch_indices = torch.full((boxes.shape[0], 1), batch_idx, dtype=boxes.dtype, device=boxes.device)
        # Concatenate batch index, boxes, and labels (unsqueeze labels to make it [num_boxes, 1])
        target_with_idx = torch.cat([batch_indices, boxes, labels.unsqueeze(1)], dim=1)
        target_list.append(target_with_idx)
        #print(target_with_idx)
    
    # Concatenate all targets along the first dimension
    return torch.cat(target_list, dim=0)

def targets_to_ultra_tensor(targets):
    target_batch = []
    for batch_idx, target in enumerate(targets):
        boxes = target['boxes']  # shape: [num_boxes, 4]
        boxes = boxes.unsqueeze(0)
        print(boxes[0])
        labels = target['labels']  # shape: [num_boxes]
        labels = labels.unsqueeze(0)
        print(labels.shape)

#@hydra.main(config_path="../../config", config_name="config")
def main():
    class_names = ['person',  'bicycle', 'motorcycle', 'vehicle']
    # Creating a model and defining number of classes
    model = yolo_v8_n(num_classes=4).cuda()
    #train_dataset, val_dataset, test_dataset = load_datasets(args)
    # model.eval()
    #torch.manual_seed(334454)

    object = 'Test.json'
    object_path = 'harborfrontv2/' + object
    output_path = 'outputs/' + object
    stupid_patth = r'/home/nieb/Projects/Big Data/Images/Seasons_drift/v2/harborfrontv2/Valid.json'
    data_dir = r'/home/nieb/Projects/Big Data/Images/Seasons_drift/v2/harborfrontv2/'
    train_dataset = COCODataset(root=data_dir, annotation=stupid_patth, numClass=4)
    train_loader = data.DataLoader(train_dataset, batch_size=16, collate_fn=collateFunction)




    epochs = 10
    with open(os.path.abspath(r'/home/nieb/Projects/DAKI Mini Projects/MLOps/model/YoloV8/args.yaml')) as f:
        params = yaml.safe_load(f)
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    #criterion = YoloCriterion(params, model)
    #criterion = ComputeLoss(model, params) # Loss fra yolo github virker, men giver infinite loss. Umiddelbart fordi targets er forkert format
    criterion = v8DetectionLoss(model, params)
    model.train()
    
    epochlosses = []
    for epoch in range(epochs):
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            #targets = targets_to_tensor(targets)
            targets = targets_to_ultra_tensor(targets)
            inputs = inputs.cuda()
            targets = targets.cuda()
            outputs = model(inputs)
            if batch_idx >=119 and batch_idx <= 121:
                print(outputs)
                print(f" targets: {targets}")
            if batch_idx > 0:
                #loss, losses = criterion(outputs, targets, batch_idx) 
                print(targets.shape)
                loss = criterion(outputs, targets)

                loss.backward()
            
                optimizer.step()
                scheduler.step()

                if batch_idx % 10 == 0:
                    print(f"batch: {batch_idx}/{len(train_loader)}: {loss.item()}")
                    #print(f"cls: {losses[0]}, box: {losses[1]}, dfl: {losses[2]}")
            # if batch_idx == 120:
            #     print(targets)
            #     print(f"For batch: {batch_idx}, loss is: {loss.item()}")
            # if batch_idx > 130:
            #     break
            # if batch_idx == 200:
            #     visualize(inputs, outputs, class_names)

            """
            if batch_idx == 120:
                img_tensor = inputs.squeeze(0).squeeze(0)
                # Scale values to [0, 255] and convert to uint8
                img_tensor = (img_tensor * 255).clamp(0, 255).to(torch.uint8)
                # Convert to NumPy array and create a PIL image in grayscale mode ('L')
                np_img = img_tensor.cpu().numpy()
                pil_img = Image.fromarray(np_img, mode='L')

                # --- Define your bounding boxes and labels ---
                # Your provided data as a tuple with a dictionary inside:
                bboxes_data = targets
                # --- Convert normalized bbox coordinates to pixel coordinates and draw ---
                # Get image dimensions
                img_width, img_height = 384, 288 

                draw = ImageDraw.Draw(pil_img)

                # Loop over each bbox and its label
                for box, label in zip(bboxes_data[0]['boxes'], bboxes_data[0]['labels']):
                    # Assuming box format: [center_x, center_y, width, height] in normalized coordinates (0 to 1)
                    cx, cy, w, h = box.tolist()
                    
                    # Convert to pixel coordinates:
                    x_min = (cx - w/2) * img_width
                    y_min = (cy - h/2) * img_height
                    x_max = (cx + w/2) * img_width
                    y_max = (cy + h/2) * img_height

                    # Draw rectangle with red outline
                    draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
                    # Optionally, draw the label as text (in yellow) at the top-left corner of the box
                    draw.text((x_min, y_min), str(label.item()), fill="yellow")

                # Display the final image with boxes drawn on it
                pil_img.show()
                break
            """
        epochlosses.append(loss)
    print(epochlosses)

    


if __name__ == "__main__":
    main()