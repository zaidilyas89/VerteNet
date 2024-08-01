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

folder1 = './New400'
folder2 = './Scans'
dst_folder = './combined'
folder_paths1 = glob(os.path.join(folder1,'*'))
folder_paths2 = glob(os.path.join(folder2,'*'))
for file_path in folder_paths1:
    fname = file_path.split('\\')[-1]
    src = os.path.join(file_path,'dxa_lateral.dcm')
    dst = os.path.join(dst_folder, fname+'.dcm')
    # pixel_array = dicom_data.pixel_array
    # img = (pixel_array - 0)/(4095-0)*255
    copy(src,dst)


for file_path in folder_paths2:
    fname = file_path.split('\\')[-1]
    src = os.path.join(file_path,'dxa_lateral.dcm')
    dst = os.path.join(dst_folder, fname+'.dcm')
    # pixel_array = dicom_data.pixel_array
    # img = (pixel_array - 0)/(4095-0)*255
    copy(src,dst)