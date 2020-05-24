# -*- coding: utf-8 -*-
"""
Resize images and corresponding bounding boxes
with Albumentations library
"""

#import libraries
import albumentations as A
from albumentations import Compose, Resize
import os
import pandas as pd
from skimage import io
import csv

#folder from which we will take images to resize
orig_fold = 'PATH'
#folder to save images after resizing them
dest_fold = 'PATH'

#load our bounding boxes as well as corresponding images' names
bboxes_data = pd.read_csv('PATH\\ingredients_bboxes.csv')
bounding_boxes = bboxes_data.iloc[:,[1,2,3,4,7]].values
img_names = bboxes_data.iloc[:,[0]].values

#define parameters for bounding boxes
#'coco' format of bbox - [xmin, ymin, w, h]
#all pictures are the same class, so we have only one label_filed
bb_params = A.BboxParams(format='coco', label_fields = ['ingr'])

#compose an augmentation (only Resize in our case) with albumentations library
def augmentation_func():
   return Compose([Resize(1152,864)], bbox_params = bb_params)

resize_the_image = augmentation_func()

#a list to store our resized bboxes
resized_bboxes = []

#take one img at a time not to run out of memory
for i, img in enumerate(img_names):
    
    #load the image
    img_file = os.path.join(orig_fold, img[0])
    img_to_resize = io.imread(img_file)
    
    #to make sure that x_max and y_max do not exceed the size of the image (it can be 0.0000000000001 for ex.)
    if bounding_boxes[i][0] + bounding_boxes[i][2] >= 3456 :
        bounding_boxes[i][2] = bounding_boxes[i][2] - 0.05
        
    if bounding_boxes[i][1] + bounding_boxes[i][3] >= 4608 :
        bounding_boxes[i][3] = bounding_boxes[i][3] - 0.05    
    
    #define the data and resize our image and corresponding bboxes
    data = {'image': img_to_resize, 'bboxes': [bounding_boxes[i]], 'ingr': [1]}
    resized_dict = resize_the_image(**data)
    
    #append bboxes to the list to save later
    resized_bboxes.append(resized_dict['bboxes'])
    
    #save the image
    resized_img = os.path.join(dest_fold, img[0])
    io.imsave(resized_img, resized_dict['image'])
    
    print(i)
    
#save resized bboxes to csv file
    
label = resized_bboxes[0][0][4] #the same for every img

#important to specify encoding, because default is cp949
with open('ingredients_resized.csv', 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['image','xmin','ymin','w','h','xmax','ymax','label'])
    for i in range (len(resized_bboxes)):
        csv_writer = csv.writer(csvfile)
        xmin = resized_bboxes[i][0][0]
        ymin = resized_bboxes[i][0][1]
        w = resized_bboxes[i][0][2]
        h = resized_bboxes[i][0][3]
        xmax = xmin + w
        ymax = ymin + h
        csv_writer.writerow([img_names[i][0], xmin,ymin,w,h,xmax,ymax,label])
   