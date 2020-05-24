# -*- coding: utf-8 -*-
"""
Convert jpg files into png
"""

import cv2
import os

#Original folder
fold = 'PATH'
#Destination folder
dest_fold = 'PATH'

#convert from jpg to png
for i, filename in enumerate(os.listdir(fold)):
    
    img1 = cv2.imread(filename)
    desttt = dest_fold + filename[:-3] + 'png'
    cv2.imwrite(desttt, img1)