# -*- coding: utf-8 -*-
"""
Created on Mon May 27 10:36:43 2024

@author: zaidi
"""

import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import os
from scipy.interpolate import CubicSpline
from custom_utilities import *

BLACKREGION_THRESHOLD = 20
def check_and_correct_increasing_sequence(x):
    corrections = {}
    for idx in range(len(x)):
        if idx == 0:
            x[idx] = x[idx]
            corrections[idx] = False
        else:
            if x[idx] <= x[idx-1]:  # Check if sequence is non-increasing
                corrections[idx] = True
                x[idx] = (x[idx + 1] + x[idx - 1]) // 2 if idx > 0 else x[idx - 1] + 1
            else:
                corrections[idx] = False
    return corrections, x

def find_location_of_cropping(x_mid, y_mid, x_dense, y_dense, width_cropped_indices):
    locations = {'crop_L1_perc':[],
                 'crop_L2_perc':[],
                 'crop_L3_perc':[],
                 'crop_L4_perc':[]}
    elements_between_L1_L2 = [x for x in width_cropped_indices if y_mid[0] <= x < y_mid[3]]
    elements_between_L2_L3 = [x for x in width_cropped_indices if y_mid[3] <= x < y_mid[6]]
    elements_between_L3_L4 = [x for x in width_cropped_indices if y_mid[6] <= x < y_mid[9]]
    elements_between_L4_L5 = [x for x in width_cropped_indices if y_mid[9] <= x < y_mid[12]]
    locations['crop_L1_perc'] = np.round(len(list(set(elements_between_L1_L2)))/(y_mid[3] - y_mid[0])*100,2)
    locations['crop_L2_perc'] = np.round(len(list(set(elements_between_L2_L3)))/(y_mid[6] - y_mid[3])*100,2)
    locations['crop_L3_perc'] = np.round(len(list(set(elements_between_L3_L4)))/(y_mid[9] - y_mid[6])*100,2)
    locations['crop_L4_perc'] = np.round(len(list(set(elements_between_L4_L5)))/(y_mid[12] - y_mid[9])*100,2)
    
    return locations




def aorta_clip_detection(img, img_with_IVGs, pts_detailed, img_name, FACTOR = 0, check_region_clipping = False, spline_save_folder = None, dpi = None, data1 = {}):
    border_clip_flag = 0 
    region_clip_flag = 0
    data = np.array([pts_detailed['Mean_L1_Left'], pts_detailed['Mean_L1_Right'], 
            pts_detailed['Mean_L2_Left'], pts_detailed['Mean_L2_Right'],
            pts_detailed['Mean_L3_Left'], pts_detailed['Mean_L3_Right'],
            pts_detailed['Mean_L4_Left'], pts_detailed['Mean_L4_Right'],
            pts_detailed['middle_1_left_L1_L2'], pts_detailed['middle_1_right_L1_L2'],
            pts_detailed['middle_2_left_L1_L2'], pts_detailed['middle_2_right_L1_L2'],
            pts_detailed['middle_1_left_L2_L3'], pts_detailed['middle_1_right_L2_L3'],
            pts_detailed['middle_2_left_L2_L3'], pts_detailed['middle_2_right_L2_L3'],
            pts_detailed['middle_1_left_L3_L4'], pts_detailed['middle_1_right_L3_L4'],
            pts_detailed['middle_2_left_L3_L4'], pts_detailed['middle_2_right_L3_L4'],
            pts_detailed['middle_1_left_L4_L5'], pts_detailed['middle_1_right_L4_L5'],
            pts_detailed['middle_2_left_L4_L5'], pts_detailed['middle_2_right_L4_L5'],
            ]).reshape(-1,4)
    
    mean_vertebral_width = np.linalg.norm((data[:,0:2]-data[:,2:]),axis=1).mean()
    # region_clip_flag = check_black_or_white_regions(data, FACTOR, img, pts_detailed, mean_vertebral_width, img_name, spline_save_folder, dpi,data1)

    spline_img = img_with_IVGs.copy()
    crop_percentage_img = img.copy()
    
    n = ['Mean_L1_Right','middle_1_right_L1_L2', 'middle_2_right_L1_L2',
         'Mean_L2_Right','middle_1_right_L2_L3', 'middle_2_right_L2_L3',
         'Mean_L3_Right','middle_1_right_L3_L4', 'middle_2_right_L3_L4',
         'Mean_L4_Right','middle_1_right_L4_L5', 'middle_2_right_L4_L5',
         'Mean_L5_Right']
    
    m = ['p1', 'p_middle_1_L1_L2', 'p_middle_2_L1_L2',
         'p2', 'p_middle_1_L2_L3', 'p_middle_2_L2_L3',
         'p3', 'p_middle_1_L3_L4', 'p_middle_2_L3_L4',
         'p4', 'p_middle_1_L4_L5', 'p_middle_2_L4_L5',
         'p5']
    

    x_mid = []
    y_mid = []
    flag_z = 0
    for i in range(0,13):
        x = pts_detailed[n[i]][0]+(mean_vertebral_width*FACTOR)
        y = int(pts_detailed[m[i]](x))
        x_mid.append(x)
        y_mid.append(y)
        cv2.circle(spline_img, (int(x), int(y)), 2, (255 * 0, 255 * 0, 255 * 1), -1, 1)
    yy = y_mid.copy()
    corrections, y_mid = check_and_correct_increasing_sequence(y_mid)
    # print(yy)
    # print(y_mid)
    
    cs = CubicSpline(y_mid, x_mid)
    
    y_dense = np.linspace(np.array(y_mid).min(), np.array(y_mid).max(), 500)
    x_dense = cs(y_dense)
    
    # Width based crop detection
    cropped = x_dense >= img.shape[1]
    if cropped.sum() > 0:
        flag_z = True
        print("Image width based crop detected")
    else:
        flag_z = False
        
    width_crop_percentage = np.round(cropped.sum()*100/len(cropped),2) 
    width_cropped_indices = []
    
    for xx, yy in zip(x_dense,y_dense):
        if xx >= img.shape[1]:
            width_cropped_indices.append(int(yy))
    
    width_locations = find_location_of_cropping(x_mid, y_mid, x_dense, y_dense, width_cropped_indices)
    
    # Black Region based crop detection
    if not flag_z:
        dicta = {}
        for j in range(0,len(x_dense)):
            value = spline_img[int(y_dense[j]),int(x_dense[j])]
            dicta[int(y_dense[j])] = value.mean()
    
        y_cropped = []
        for key, value in dicta.items():
            if value <=BLACKREGION_THRESHOLD:
                y_cropped.append(key)
            
    
        if len(y_cropped) > 0:
            black_region_crop_percentage = np.round(len(set(y_cropped))*100/len(y_dense),2)
        else:
            black_region_crop_percentage =  np.round(0.0 *100,2)
    else:
        black_region_crop_percentage =  np.round(0.0 *100,2)
        y_cropped = []
        
    # Location of Cropping
    
    black_locations = find_location_of_cropping(x_mid, y_mid, x_dense, y_dense, y_cropped)
    if black_region_crop_percentage > 0:
        border_flag = 1
    else:
        border_flag = 0      
    
    # print(width_crop_percentage)
    # print(black_region_crop_percentage)
    # try:
    #     points = []
    #     for j in range(0,len(x_dense)):
    #         points.append(spline_img[int(y_dense[j]),int(x_dense[j])])
    #         cv2.circle(spline_img, (int(x_dense[j]), int(y_dense[j])), 2, (0, 0, 255), -1)

    # except IndexError as e:
    #     # Check if the error message indicates that the index is out of bounds for axis 1
    #     if 'index' in str(e) and 'is out of bounds for axis 1' in str(e):
    #     # Handle the specific IndexError for axis 1 here
    #         print("Index is out of bounds for axis 1")

    X=600
    OFFSET = 80
    plt.imshow(spline_img)
    plt.scatter(x_mid, y_mid, color='red', s=3, label='Original Pixels')
    plt.plot(x_dense, y_dense, color='blue', label='Cubic Spline Fit')
    # plt.text(X, 50, f'Blk_Reg Perc: {black_region_crop_percentage}', color='red', fontsize=12, 
    #      bbox=dict(facecolor='white', alpha=0.5))
    # plt.text(X, 50 + OFFSET, f"Blk_L1 Perc: {black_locations['crop_L1_perc']}", color='red', fontsize=12, 
    #      bbox=dict(facecolor='white', alpha=0.5))
    # plt.text(X,  50 + 2*OFFSET, f"Blk_L2 Perc: {black_locations['crop_L2_perc']}", color='red', fontsize=12, 
    #      bbox=dict(facecolor='white', alpha=0.5))
    # plt.text(X, 50 + 3*OFFSET, f"Blk_L3 Perc: {black_locations['crop_L3_perc']}", color='red', fontsize=12, 
    #      bbox=dict(facecolor='white', alpha=0.5))
    # plt.text(X, 50 + 4*OFFSET, f"Blk_L4 Perc: {black_locations['crop_L4_perc']}", color='red', fontsize=12, 
    #      bbox=dict(facecolor='white', alpha=0.5))
    plt.text(X, 50 + 5*OFFSET, f'Bord_Perc: {width_crop_percentage}', color='blue', fontsize=12, 
         bbox=dict(facecolor='white', alpha=0.5))
    plt.text(X, 50 + 6*OFFSET, f"Bord_L1 Perc: {width_locations['crop_L1_perc']}", color='blue', fontsize=12, 
         bbox=dict(facecolor='white', alpha=0.5))
    plt.text(X,  50 + 7*OFFSET, f"Bord_L2 Perc: {width_locations['crop_L2_perc']}", color='blue', fontsize=12, 
         bbox=dict(facecolor='white', alpha=0.5))
    plt.text(X, 50 + 8*OFFSET, f"Bord_L3 Perc: {width_locations['crop_L3_perc']}", color='blue', fontsize=12, 
         bbox=dict(facecolor='white', alpha=0.5))
    plt.text(X, 50 + 9*OFFSET, f"Bord_L4 Perc: {width_locations['crop_L4_perc']}", color='blue', fontsize=12, 
         bbox=dict(facecolor='white', alpha=0.5))
    # plt.text(X, 50 + 10*OFFSET, f"Black_Pixel_Threshold: {BLACKREGION_THRESHOLD}", color='blue', fontsize=12, 
    #      bbox=dict(facecolor='white', alpha=0.5))

    
    plt.savefig(os.path.join(spline_save_folder,'Aorta_Clip_Flag_'+'Border_Crop_'+str(border_flag)+'_Region_Crop_'+str(flag_z*1)+'_'+img_name), dpi=dpi)
    plt.close()


    
    
    
    
    
    return border_clip_flag, region_clip_flag

