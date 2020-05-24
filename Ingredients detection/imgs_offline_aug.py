# -*- coding: utf-8 -*-
"""
Augment images for offline augmentation with scikit-image
"""

#Import libraries
import pandas as pd
import csv
import imgaug as ia
import imgaug.augmenters as iaa
from skimage import io
import os
import numpy as np

#folder from which we will take images to resize
orig_fold = 'PATH'
#folder to save images after resizing them
dest_fold = 'PATH'

#load our bounding boxes as well as corresponding images' names
bboxes_data = pd.read_csv('PATH')
bounding_boxes = bboxes_data.iloc[:,[5,6,9,10]].values
img_names = bboxes_data.iloc[:,[0]].values

#define augmentations of two kinds
augment_1 = iaa.Sequential([iaa.PiecewiseAffine(scale = 0.035), iaa.MultiplySaturation(mul = 0.75)]) 

augment_2 = iaa.Sequential([
                        iaa.MotionBlur(k=3),
                        iaa.MultiplyAndAddToBrightness(mul = 1, add = 30),
                        iaa.MultiplySaturation(mul = 2),
                        iaa.AdditiveGaussianNoise(scale = 3)
                       ])


#lists for augmentations of bounding boxes
aug_1_bb_list = []
aug_2_bb_list = []

#augment images with two kinds of augmentations
for i, img in enumerate(img_names):
    
    #get an image
    img_file = os.path.join(orig_fold, img[0])
    img_to_aug = io.imread(img_file)
    img_to_aug_exp = np.expand_dims(img_to_aug, axis = 0)
    
    #define bounding boxes
    xmin = bounding_boxes[i][0]
    ymin = bounding_boxes[i][1]
    xmax = bounding_boxes[i][2]
    ymax = bounding_boxes[i][3]
    
    #skimage bounding box
    b_box = [[ia.BoundingBox(x1 = xmin,y1 = ymin,x2 = xmax,y2 = ymax)]]
    
    #augment images and corresponding bounding boxes
    img_aug1, bbox_aug1  = augment_1(images = img_to_aug_exp, bounding_boxes = b_box)
    img_aug2, bbox_aug2  = augment_2(images = img_to_aug_exp, bounding_boxes = b_box)
    
    #append augmentedcoordinates of bounding boxes to lists
    aug_1_bb_list.append(['res1_'+img[0], bbox_aug1[0][0][0][0], bbox_aug1[0][0][0][1],
                          bbox_aug1[0][0][1][0], bbox_aug1[0][0][1][1]])
    
    aug_2_bb_list.append(['res2_'+img[0], bbox_aug2[0][0][0][0], bbox_aug2[0][0][0][1],
                          bbox_aug2[0][0][1][0], bbox_aug2[0][0][1][1]])
    
    #save images
    res_img_1 = os.path.join(dest_fold, 'res1_'+img[0])
    res_img_2 = os.path.join(dest_fold, 'res2_'+img[0])
    
    io.imsave(res_img_1, img_aug1[0])
    io.imsave(res_img_2, img_aug2[0])
    
    
#create csv files
#important to specify encoding, because default is cp949
with open('ingredients_train_aug1.csv', 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['image','xmin','ymin','xmax','ymax'])
    for i in range (len(aug_1_bb_list)):
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(aug_1_bb_list[i])
        
with open('ingredients_train_aug2.csv', 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['image','xmin','ymin','xmax','ymax'])
    for i in range (len(aug_2_bb_list)):
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(aug_2_bb_list[i])