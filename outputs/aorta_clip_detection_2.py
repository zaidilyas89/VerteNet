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

def spline_fit_check(n, m, img, img2, mean_vertebral_width,FACTOR, pts_detailed, img_name):
    x_mid = []
    y_mid = []
    
    for i in range(0,13):
        x = pts_detailed[n[i]][0]+(mean_vertebral_width*FACTOR)
        y = int(pts_detailed[m[i]](x))
        x_mid.append(x)
        y_mid.append(y)
        cv2.circle(img, (int(x), int(y)), 2, (255 * 0, 255 * 0, 255 * 1), -1, 1)
  
    cs = CubicSpline(y_mid, x_mid)
    
    y_dense = np.linspace(np.array(y_mid).min(), np.array(y_mid).max(), 500)
    x_dense = cs(y_dense)
    try:
        points = []
        for j in range(0,len(x_dense)):
            points.append(img2[int(y_dense[j]),int(x_dense[j])])
            flag_z = 0
        
    except IndexError as e:
        # Check if the error message indicates that the index is out of bounds for axis 1
        if 'index' in str(e) and 'is out of bounds for axis 1' in str(e):
        # Handle the specific IndexError for axis 1 here
            print("Index is out of bounds for axis 1")
            flag_z = 1
    # else:
    #     print('hey!')
    #     # If it's not the specific IndexError we're looking for, re-raise the exception
    #     raise

    
    plt.imshow(img)
    plt.scatter(x_mid, y_mid, color='red', s=3, label='Original Pixels')
    plt.plot(x_dense, y_dense, color='blue', label='Cubic Spline Fit')
    plt.show()
    
    return x_mid, y_mid, points, flag_z


def check_black_or_white_regions(data, FACTOR, img, pts_detailed, mean_vertebral_width, img_name):
    img2 = img.copy()
    
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
    
    # FACTOR = 0.5
    x_mid_1, y_mid_1, points_1, flag_z_1 = spline_fit_check(n, m, img, img2, mean_vertebral_width,FACTOR, pts_detailed, img_name)
    # x_mid_2, y_mid_2, points_2 = spline_fit_check(n, m, img, img2, mean_vertebral_width,FACTOR+0.15, pts_detailed, img_name)
    # x_mid_3, y_mid_3, points_3 = spline_fit_check(n, m, img, img2, mean_vertebral_width,FACTOR-0.15, pts_detailed, img_name)

    # points = np.concatenate([np.array(points_1), np.array(points_2), np.array(points_3)],1)
    points = np.array(points_1)
    if flag_z_1 > 1:
        flag2 = 1
    else:
        flag2 = 0
        
    plt.savefig(os.path.join('./splines','Region_Clip_Flag_'+str(flag2)+'_'+img_name))
    plt.close()
    return flag2


def aorta_clip_detection(img, img_with_IVGs, pts_detailed, img_name, FACTOR = 0.75, check_region_clipping = False):
    
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

    h,w,c = img.shape
    region_clip_flag = 0
    left_clip = np.array((w - (data+(mean_vertebral_width*FACTOR)))[:,0])
    if (left_clip<0).sum() > 0:
        left_clip_flag = 1
    else: left_clip_flag = 0
    right_clip = np.array((w - (data+(mean_vertebral_width*FACTOR)))[:,2])
    if  (right_clip<0).sum() > 0:  
        right_clip_flag = 1
    else: right_clip_flag = 0
    if left_clip_flag == 1 or right_clip_flag == 1:
        # print('Aorta Clip Detected')
        # print('check')
        border_clip_flag = 1
    else:
        border_clip_flag = 0
        
    if border_clip_flag == 0 and check_region_clipping == True:

        try:
            region_clip_flag = check_black_or_white_regions(data, FACTOR, img, pts_detailed, mean_vertebral_width, img_name)
            
        except Exception:
        # Ignore the exception and continue
            print('Here')
            # region_clip_flag = check_black_or_white_regions(data, FACTOR, img, pts_detailed, mean_vertebral_width, img_name)
            region_clip_flag = 1
            pass
    return border_clip_flag, region_clip_flag


# # -*- coding: utf-8 -*-
# """
# Created on Mon May 27 10:36:43 2024

# @author: zaidi
# """

# import pandas as pd
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np




# def aorta_clip_detection(img, img_with_IVGs, pts_detailed, FACTOR = 0.75):
    
#     data = np.array([pts_detailed['Mean_L1_Left'], pts_detailed['Mean_L1_Right'], 
#             pts_detailed['Mean_L2_Left'], pts_detailed['Mean_L2_Right'],
#             pts_detailed['Mean_L3_Left'], pts_detailed['Mean_L3_Right'],
#             pts_detailed['Mean_L4_Left'], pts_detailed['Mean_L4_Right'],
#             pts_detailed['middle_1_left_L1_L2'], pts_detailed['middle_1_right_L1_L2'],
#             pts_detailed['middle_2_left_L1_L2'], pts_detailed['middle_2_right_L1_L2'],
#             pts_detailed['middle_1_left_L2_L3'], pts_detailed['middle_1_right_L2_L3'],
#             pts_detailed['middle_2_left_L2_L3'], pts_detailed['middle_2_right_L2_L3'],
#             pts_detailed['middle_1_left_L3_L4'], pts_detailed['middle_1_right_L3_L4'],
#             pts_detailed['middle_2_left_L3_L4'], pts_detailed['middle_2_right_L3_L4'],
#             pts_detailed['middle_1_left_L4_L5'], pts_detailed['middle_1_right_L4_L5'],
#             pts_detailed['middle_2_left_L4_L5'], pts_detailed['middle_2_right_L4_L5'],
#             ]).reshape(-1,4)
    
#     mean_vertebral_width = np.linalg.norm((data[:,0:2]-data[:,2:]),axis=1).mean()

#     h,w,c = img.shape

#     left_clip = np.array((w - (data+(mean_vertebral_width*FACTOR)))[:,0])
#     if (left_clip<0).sum() > 0:
#         left_clip_flag = 1
#     else: left_clip_flag = 0
#     right_clip = np.array((w - (data+(mean_vertebral_width*FACTOR)))[:,2])
#     if  (right_clip<0).sum() > 0:  
#         right_clip_flag = 1
#     else: right_clip_flag = 0
#     if left_clip_flag == 1 or right_clip_flag == 1:
#         # print('Aorta Clip Detected')
#         # print('check')
#         flag = 1
#     else:
#         flag = 0
#     return flag