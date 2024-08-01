# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 08:22:50 2024

@author: zaidi
"""

import os
import pandas as pd
import numpy as np
from glob import glob
from shutil import copy
from custom_utilities import *
from tqdm import tqdm
base_path_file = r'D:\12_AAC_Semantic_Segmentation\Landmark Detection\outputs\us_model'
file_1 = 'id_to_labels_model.csv'
file_2 = 'id_to_labels_us.csv'
dst_base_path_file = r'D:\12_AAC_Semantic_Segmentation\Landmark Detection\outputs\aac_greater_than'






create_folder(dst_base_path_file)


numbers = np.round(np.arange(0.5, 1.6, 0.1),1)
factors = numbers.tolist()
factor_folders = []
for fctr in factors:
    factor_folder = os.path.join(dst_base_path_file,'Factor_'+str(fctr))
    create_folder(factor_folder)
    factor_folders.append(factor_folder)
    

df1 = pd.read_csv(os.path.join(base_path_file, file_1))
df2 = pd.read_csv(os.path.join(base_path_file, file_2))

df3 = pd.concat((df1,df2),axis = 0)

df4 = df3[df3['AAC_Score']>=3]

fnames = df4['Scan_with_Date'].tolist()


for folder in tqdm(factor_folders):
    factor = folder.split('\\')[-1]
    src_folder = os.path.join(base_path_file,'splines',factor)
    files = glob(os.path.join(src_folder,'*.png'))
    f = []
    g = []
    for i in tqdm(files):
        
        f.append(i.split('\\')[-1].split('Aorta_Clip_Flag_0_')[-1].split('Aorta_Clip_Flag_1_')[-1].split('.png')[0])
        a = df4[df4['Scan_with_Date'].isin(f)]
        
        for fname in tqdm(a['Scan_with_Date']):
            src_path = glob(os.path.join(src_folder,'*'+fname+'*'))
            n = src_path[0].split('\\')[-1]
            dst_path = os.path.join(folder,n.split('.png')[0]+'_AAC_scoe_'+str(a[a['Scan_with_Date']==fname]['AAC_Score'].item())+'.png')
            try:
                copy(src_path[0],dst_path)
            except:
                continue