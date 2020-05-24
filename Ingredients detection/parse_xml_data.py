# -*- coding: utf-8 -*-
"""
Created on Tue May 12 17:24:16 2020

@author: Alena
"""

import xml.etree.ElementTree as ET
import os
import csv
import pandas as pd
import shutil

#folder
fold = 'D:\\Programming\\PORTFOLIO\\Text_rec\\oxford_pet_dataset\\doggos_xml'
files_list = os.listdir(fold)


def read_xml_file (xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        for boxes in root.iter('object'):
            
            filename = root.find('filename').text
            
            for box in boxes.findall('bndbox'):
                xmin = int(box.find('xmin').text)
                ymin = int(box.find('ymin').text)
                xmax = int(box.find('xmax').text)
                ymax = int(box.find('ymax').text)
                
            return [filename, xmin, ymin, xmax, ymax, 'doggo']

    
    
with open('doggos_bboxes.csv', 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['image','xmin','ymin','xmax','ymax','label'])
    for xml_filename in files_list:
        csv_writer = csv.writer(csvfile)
        xml_file = os.path.join(fold,xml_filename)
        temp = read_xml_file(xml_file)
        csv_writer.writerow(temp)
        
        
        
#Separate images which have bbox and which don't
orig_fold = 'D:\\Programming\\PORTFOLIO\\Text_rec\\oxford_pet_dataset\\doggos'
dest_fold = 'D:\\Programming\\PORTFOLIO\\Text_rec\\oxford_pet_dataset\\doggos_with_bboxes'

bboxes = pd.read_csv('doggos_bboxes.csv')
files_to_move = bboxes.iloc[:,0].values

for img in files_to_move:
    orig = os.path.join(orig_fold,img)
    dest = os.path.join(dest_fold,img)
    shutil.move(orig,dest)
