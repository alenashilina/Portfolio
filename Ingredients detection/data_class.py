# -*- coding: utf-8 -*-
"""
Creating dataclass following pytorch tutorial
"""

#Import libraries
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import skimage
from skimage import io
import os

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


#Create the class
class IngredientsDataset(Dataset):
    """Dataset for food lables with bounding boxes for ingredients"""

    #Create __init__ with args: self, csv_file, root for directory with imgs, and transformation
    def __init__(self, csv_file, img_folder, transform=None):
        self.boundary_box_file = pd.read_csv(csv_file)
        self.img_folder = img_folder
        self.transform = transform
    
    #Create __len__ with arg: self
    def __len__(self):
        return len(self.boundary_box_file)
    
    #Create __getitem__ with args: self, index
    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        #getting the image
        img_file = os.path.join(self.img_folder, self.boundary_box_file.iloc[idx,0])
        img_idx = io.imread(img_file)
        img_float = skimage.img_as_float64(img_idx)
        
        #getting the normalized bounding box: x_min, y_min, x_max, y_max
        bounding_box = self.boundary_box_file.iloc[idx, 1:5]
        bounding_box = np.array([bounding_box])            
        bounding_box = bounding_box.astype('float').reshape(-1,4)
        
        #transform if needed
        if self.transform is not None:
            img_float = self.transform(img_float)
        
        #get the dictionary with the image and its bounding box
        sample = {'image': img_float, 'bounding box': bounding_box}
        
                
        return sample
        
        