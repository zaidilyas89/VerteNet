# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 12:19:01 2024

@author: zaidi
"""


import os 
import pandas as pd
from glob import glob
from pydicom import dcmread
from shutil import copy
from tqdm import tqdm
import argparse

# Create the parser
parser = argparse.ArgumentParser(description="CLSA DXA images extraction from folders.")

# Add an argument for the folder path
parser.add_argument('--src_folder_path', 
                    type=str, nargs='?',  
                    default = './23SP004_F1',  
                    help='Enter Source Folder.')
parser.add_argument('--dst_folder_path', 
                    type=str, nargs='?',  
                    default = './dcm',  
                    help='Enter Destination Folder.')

# Parse the arguments
args = parser.parse_args()

# Use the folder path argument
folder_path = args.src_folder_path
dst_folder = args.dst_folder_path

folder1 = folder_path
# folder2 = './Scans'


# Check if the folder exists
if not os.path.exists(dst_folder):
    # Create the folder if it doesn't exist
    os.makedirs(dst_folder)
    print(f"Folder created: {dst_folder}")
else:
    print(f"Folder already exists: {dst_folder}")

folder_paths1 = glob(os.path.join(folder1,'*'))
# folder_paths2 = glob(os.path.join(folder2,'*'))
for file_path in tqdm(folder_paths1):
    fname = file_path.split('\\')[-1]
    src = os.path.join(file_path,'dxa_lateral.dcm')
    dst = os.path.join(dst_folder, fname+'.dcm')
    # pixel_array = dicom_data.pixel_array
    # img = (pixel_array - 0)/(4095-0)*255
    copy(src,dst)


# for file_path in folder_paths2:
#     fname = file_path.split('\\')[-1]
#     src = os.path.join(file_path,'dxa_lateral.dcm')
#     dst = os.path.join(dst_folder, fname+'.dcm')
#     # pixel_array = dicom_data.pixel_array
#     # img = (pixel_array - 0)/(4095-0)*255
#     copy(src,dst)
