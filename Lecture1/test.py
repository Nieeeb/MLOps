import os
import sys
root_dir = os.getcwd()
sys.path.append(root_dir)

from image_dataloder import ThermalDataset
from model.detr import DETR
from model.backbone import build_backbone
import pandas as pd

backbone = build_backbone()
model = DETR(backbone=backbone, num_classes=1)

images_path = r"Data/training/Feb Day"
metadata_file_name = r"Data/training_metadata/feb_day.csv"

metadata = pd.read_csv(metadata_file_name)

metadata["DateTime"] = pd.to_datetime(metadata['DateTime'], dayfirst = False)

dataset = ThermalDataset(img_dir=images_path, selection = metadata, return_metadata = True, check_data= True)