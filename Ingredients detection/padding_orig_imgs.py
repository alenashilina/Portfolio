# -*- coding: utf-8 -*-
"""
We have two kinds of images: horisontal (3456 x 4608) and vertical (4608 x 3456) ones.
Pad horisotal images with white color, to make them vertical.
"""

import cv2
import os

fold = 'PATH'
for i, filename in enumerate(os.listdir(fold)):

    img1 = cv2.imread(filename)
    img_width = img1.shape[1]
        
    #if the image is horizontal (3456 x 4608 - H x W)
    if img_width == 4608:
            
        #resize
        resized_img = cv2.resize(img1, (3456,2592))
            
        #pad
        WHITE = [255,255,255]
            
        #pad the image with solid white color from top and bottom
        constant = cv2.copyMakeBorder(resized_img, 1008,1008,0,0, cv2.BORDER_CONSTANT, value = WHITE)
        
        #save image
        cv2.imwrite(filename,constant)
        print(i)
        
    else:
        #needed for VoTT to process images correctly
        resized_img2 = cv2.resize(img1, (3456,4608))
        cv2.imwrite(filename,resized_img2)
            