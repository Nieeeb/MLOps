import torch
from IPython.display import Image  # for displaying images
import os 
import random
import shutil
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

random.seed(42)

data = r"Data/training/Feb Day"

def load_data(datapath):
    alldata = os.listdir(datapath)

    images = [file for file in alldata if ".jpg" in file]
    bounding_boxes = [file for file in alldata if ".txt" in file]

    return images, bounding_boxes




# A: Finde ud af hvordan YOLOv5 gerne vil have data ind - William
# B: Hvordan initialiserer man en YOLOv5 model så den er klar til at træne/så den kan få noget af dataen - Victor
# C: Plotte en bounding box på vores data - Xander


