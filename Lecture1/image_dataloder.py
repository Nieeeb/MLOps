# -*- coding: utf-8 -*-
"""
Created on Wed May  5 09:00:02 2021

@author: IvanTower
"""

import cv2
import os
from sys import platform
import numpy as np
import pandas as pd

import re

import torch
import torch.utils.data

#  Dataset class for image data depending on the metadata

'''
For initializing the class the inputs are:
    img_dir - the top directory containing all the images
    metadata - pandas dataframe containing references to the selected images
    return_metadata - boolean deciding if metadata would be returned as part of __getitem__ calls
    pattern_img - the type of extension that is being searched for
'''

class ThermalDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, selection, return_metadata = True, transforms=None, check_data = False, pattern_img = '.jpg'):

        self.selection = selection
        self.return_metadata = return_metadata
        self.transforms = transforms
        

        # file_path_list = img_dir + "/" + selection["Folder name"].astype(str) + "_" + selection["Clip Name"].astype(str) + "_img_" +  selection["Image Number"].astype(str) + pattern_img
            
        # for img in file_path_list:
        #     print(img)
        #     break

        #  create a list of image directories from the metadata entries
        if platform == "linux" or platform == "linux2" or platform == "darwin":
            self.image_list = img_dir + "/" + selection["Folder name"].astype(str) + "_" + selection["Clip Name"].astype(str) + "_img_" +  selection["Image Number"].astype(str) + pattern_img
        else:
            self.image_list = img_dir + "\\" + selection["Folder name"].astype(str) + "\\" + selection["Clip Name"].astype(str) + "\\" + "image_" +  selection["Image Number"].astype(str).str.zfill(4) + pattern_img

        # check if list is empty
        if not len(self.image_list)>0:
            print("No images found - check metadata")

        if check_data:
            # check if there are paths from the image_list where the path does not exist/incorrect
            check_path = [not os.path.exists(path) for path in self.image_list]
            # get incorrect parths
            wrong_paths = self.image_list[check_path]

            #  signal that there are incorrect paths
            if len(wrong_paths)>0:
                print(f"Warning {len(wrong_paths)} image paths do not exist!")
                print(wrong_paths)

    # function for loading a specific image and its path
    def load_image(self, image_path):
        img = cv2.imread(image_path,-1)
        cv2.imshow("hahah", img)
        return img[:, :, np.newaxis], image_path

    # function for getting the metadata for the specified image
    def get_img_metadata(self, image_path):
        if platform == "linux" or platform == "linux2" or platform == "darwin":
            img_path_split = image_path.split("_")
        else:
            img_path_split = image_path.split("\\")
        img_folder = img_path_split[0].split('/')[3]
        

        clip_file = img_path_split[1].split(".")[0] + "_" + img_path_split[2].split(".")[0] + "_" + img_path_split[3].split(".")[0]

        img_file = img_path_split[-1].split(".")[0].split("_")[0]

        # print(f" {img_folder}          {img_file}")
        curr_metadata = self.selection[(self.selection["Folder name"]== int(img_folder))&(self.selection["Clip Name"]== clip_file)&(self.selection["Image Number"]== int(img_file))].dropna()
        print(curr_metadata)
        return curr_metadata

    #  __getitem__ function that loads an image from an index, normalizes it between 0 and 1, makes it into a tensor of float and unsqueezes it
    def __getitem__(self, idx):
        img, path = self.load_image(self.image_list.iloc[idx])

        if self.transforms:
            sample = self.transforms(**{'image':img})
            img = sample["image"]

        x = img.transpose((2, 0, 1))
        x = torch.as_tensor(x, dtype=torch.float32)

        # if metadata is returned for the selected images then make the metadata into a string, because dataloaders do not like pandas dataframes
        if self.return_metadata:
            metadata_curr = self.get_img_metadata(self.image_list.iloc[idx])
            metadata_curr['DateTime'] = metadata_curr['DateTime'].dt.strftime("%Y-%m-%d %H:%M:%S")
            output_metadata_list = metadata_curr.squeeze().values.tolist()
            output_metadata_str = ','.join(map(str, output_metadata_list))

            # print(metadata_curr)

            return x, path, output_metadata_str

        return x

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':

    images_path = r"Data/training/Feb Day"
    metadata_file_name = r"Data/training_metadata/feb_day.csv"
    #metadata_path = os.path.join(images_path,metadata_file_name)

    metadata = pd.read_csv(metadata_file_name)

    metadata["DateTime"] = pd.to_datetime(metadata['DateTime'], dayfirst = False)

    # metadata_selected = metadata[(metadata["DateTime"].dt.day>=1) & (metadata["DateTime"].dt.day<=7) & (metadata["DateTime"].dt.month==2)]

    dataset = ThermalDataset(img_dir=images_path, selection = metadata, return_metadata = True, check_data= True)

    # trial = next(iter(dataset))



    
    for i, sample in enumerate(dataset):
        img, path, metadata = sample
        cv2.waitKey(0)
