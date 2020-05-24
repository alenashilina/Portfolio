# -*- coding: utf-8 -*-
"""
Devide files by folders train/val/test
"""

import os
import shutil

#folder with all images
fold = 'PATH'
#folder where to move files
destination = 'PATH'

img_list = os.listdir(fold)

train = img_list[0:671]
val = img_list[671:756]
test = img_list[756:]

#move images by folders
for i in range(len(val)):
    shutil.move(fold+val[i],destination+'val\\'+val[i])

for i in range(len(train)):
    shutil.move(fold+train[i],destination+'train\\'+train[i])
    
for i in range(len(test)):
    shutil.move(fold+test[i],destination+'test\\'+test[i])
