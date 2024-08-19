# -*- coding: utf-8 -*-
"""
Created on Mon May 27 08:54:27 2024

@author: zaidi
"""

import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from glob import glob
from PIL import Image
import numpy as np
import draw_points_detailed
import aorta_clip_detection, aorta_clip_detection_3
# from aorta_clip_detection_3 import sort_and_copy
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from shutil import copy
from custom_utilities import *
import pandas as pd
import argparse
# Create the parser
parser = argparse.ArgumentParser(description="Aorta Crop Detection Splines Results for different factors in Excel File.")

# Add an argument for the folder path
parser.add_argument('--src_folder_path', 
                    type=str, nargs='?',  
                    default = './clsa_30000/splines',  
                    help='Enter Source Folder having splines for different factors.')
parser.add_argument('--dst_folder_path', 
                    type=str, nargs='?',  
                    default = './clsa_f1_5000_detailed_splines/splines',  
                    help='Enter Destination Folder.')

# Parse the arguments
args = parser.parse_args()

# Use the folder path argument
src_folder = args.src_folder_path
dst_folder = args.dst_folder_path

folder_details = list(os.walk(src_folder))
factor_folders = folder_details[0][1]

img_names_with_flags = folder_details[-1][2]

img_names = []
for img_name_with_flag in img_names_with_flags:
    img_names.append(img_name_with_flag.split('_')[-1])    

l = len(factor_folders)

dicta = {factor_folder:[] for factor_folder in factor_folders}
dicta['img_name'] = img_names
for idx, factor_factor in enumerate(tqdm(factor_folders)):
    # base_path = folder_details[idx+1]
    img_files = folder_details[idx+1][2]
    df = pd.DataFrame(img_files, columns=['names'])
    
    for img_file in tqdm(dicta['img_name']):
        a = df[df['names'].str.contains(img_file, case=False, na=False)]
        _,_,_,flag,name = a['names'].item().split('_')
        crop_status = int(flag)
        i = dicta['img_name'].index(img_file)
        assert name == img_file
        dicta[factor_factor].append(crop_status)
        
df1 = pd.DataFrame(dicta)
df1.to_csv(os.path.join(dst_folder,src_folder.split('/')[1]+'.csv'), index=None)
