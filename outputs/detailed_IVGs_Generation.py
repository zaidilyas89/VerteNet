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
import aorta_clip_detection
data_folder = './GuideNetv3_predictions_detailed'
detailed_IVGs_folder = './detailed_IVGs'
import pandas as pd

dataset = {}
sub_folders = list(os.walk(data_folder))[0][1]



for i in tqdm(range(0, len(sub_folders)-7)):
    sub_folder = sub_folders[i]
    
    imgs_path = os.path.join(data_folder,sub_folder,'original_images')
    landmarks_path = os.path.join(data_folder,sub_folder,'landmarks')
    
    img_names = list(os.walk(imgs_path))[0][2]
    
    data_region_clip = []
    data_border_clip = []
    for img_name in tqdm(img_names):
        # if img_name != 'de_SBH-09072015_090913.png':
        #     continue
        landmark_name = img_name.split('.png')[0]+'.csv'
        img = Image.open(os.path.join(imgs_path,img_name))
        img = np.array(img)
        orig_img = img.copy()
        pts = pd.read_csv(os.path.join(landmarks_path,landmark_name),index_col=False)
        pts0 = pts.iloc[:,1:].values
        # pts_ = pts0.iloc[:,3:-1].values
        # pts0 = x1.reshape(24,2)
        
        ori_image_regress, ori_image_points, ori_image_points_points_only, pts_detailed = draw_points_detailed.draw_landmarks_regress_test(pts0,
                                                                                               img.copy(),
                                                                                               img)
        
        plt.tight_layout()
    
        plt.imshow(ori_image_points)
        plt.savefig(os.path.join(detailed_IVGs_folder,img_name), bbox_inches='tight')
        plt.close()
        border_clip_flag, region_clip_flag = aorta_clip_detection.aorta_clip_detection(orig_img, ori_image_points, pts_detailed, img_name, FACTOR = 0.75, check_region_clipping = True)
        if border_clip_flag == 1:
            data_border_clip.append(img_name)
        if region_clip_flag == 1:
            data_region_clip.append(img_name)
    dataset[sub_folder] = {'region_clip':data_region_clip,
                           'border_clip':data_border_clip}


        