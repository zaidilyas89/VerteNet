# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:34:32 2024

@author: zaidi

"""


import pandas as pd
import numpy as np
import os
from glob import glob
from shutil import copy


def create_folders_if_not_exist(main_folder, subfolders):
    try:
        os.makedirs(main_folder)
        print(f"Main folder created: {main_folder}")
    except FileExistsError:
        print(f"Main folder already exists: {main_folder}")

    for subfolder in subfolders:
        subfolder_path = os.path.join(main_folder, subfolder)
        try:
            os.makedirs(subfolder_path)
            print(f"Subfolder created: {subfolder_path}")
        except FileExistsError:
            print(f"Subfolder already exists: {subfolder_path}")
            
file = './ids.csv'
folder_orig_imgs = './GuideNetv3_predictions_detailed/caifos_detailed_outputs/original_images'
folder_landmark_imgs = './GuideNetv3_predictions_detailed/caifos_detailed_outputs/images_with_landmarks_and_IVGs'
folder_landmarks = './GuideNetv3_predictions_detailed/caifos_detailed_outputs/landmarks'

dest_folder = './new'
df = pd.read_csv(file)

main_folder = "John_Data_Analysis_PLSAW"
subfolders = ["landmarks", "original_images", "original_images_with_landmarks"]

create_folders_if_not_exist(main_folder, subfolders)



file_orig =[]
file_lm = []
for idx, fname in enumerate(df['ids']):
    
    year = df.iloc[idx,1].split('/')[-1]
    if int(year) < 2000:
        name = str(fname).zfill(4)+'_BL'
    else:
        name = str(fname).zfill(4)+'_F60'
    file_orig.extend(glob(os.path.join(folder_orig_imgs, f'*{name}*')))
    file_lm.extend(glob(os.path.join(folder_landmark_imgs, f'*{name}*')))
    if len(glob(os.path.join(folder_landmark_imgs, f'*{name}*'))) == 0:
        print(name)
    assert len(file_orig) == len(file_lm)
    

for forig,flm in zip(file_orig,file_lm):
    n = forig.split('\\')[-1]
    landmarks_file_name = n.split('.png')[0]+'.csv'
    copy(forig,os.path.join(main_folder,subfolders[-1],n))
    copy(flm,os.path.join(main_folder,subfolders[-2],n))
    
    a = os.path.join(flm.split('\\')[0].split('images_with_landmarks_and_IVGs')[0],'landmarks',landmarks_file_name)
    b = os.path.join(main_folder,subfolders[0],landmarks_file_name)
    
    copy(a,b)