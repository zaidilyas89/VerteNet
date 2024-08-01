# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:29:12 2024

@author: zaidi
"""

import os
import pandas as pd
import numpy as np
from glob import glob
from PIL import Image
import cv2
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

# from scipy.io import loadmat
path1 = './John_Data_Analysis_PLSAW\original_images'
path2 = './John_Data_Analysis_PLSAW\landmarks'
files = glob(os.path.join(path1,'*.png'))
dst = './analysis_images'




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
    points = []
    for j in range(0,len(x_dense)):
        points.append(img2[int(y_dense[j]),int(x_dense[j])])
    plt.imshow(img)
    plt.scatter(x_mid, y_mid, color='red', s=3, label='Original Pixels')
    plt.plot(x_dense, y_dense, color='blue', label='Cubic Spline Fit')
    plt.show()
    
    return x_mid, y_mid, points


ratios = []
fnames = []
for file_path in files:
    
    fname_csv = file_path.split('\\')[-1].split('.png')[0]+'.csv'
    fname_csv_path = os.path.join(path2, fname_csv)
    pred_labels_v1 = pd.read_csv(fname_csv_path,index_col=False)
    x1 = pred_labels_v1.iloc[:,3:-1].values
    x_v1 = x1.reshape(24,2)
    data = x_v1.reshape(12,4)
    img = Image.open(file_path)
    img2 = np.array(img)
    w, h = img.size
    fnames.append(file_path.split('\\')[-1])
    
    
    mean_vertebral_width = np.linalg.norm((data[:,0:2]-data[:,2:]),axis=1).mean()
    max_vertebral_width = np.linalg.norm((data[:,0:2]-data[:,2:]),axis=1).max()
    min_vertebral_width = np.linalg.norm((data[:,0:2]-data[:,2:]),axis=1).min()
    median_vertebral_width = np.median(np.linalg.norm((data[:,0:2]-data[:,2:]),axis=1))
    
    try:
        cs = CubicSpline(data[:,3], data[:,2])
    except:
        continue
    
    y_dense = np.linspace(np.array(data[:,3]).min(), np.array(data[:,3]).max(), 500)
    x_dense = cs(y_dense)
    # points = []
    # for j in range(0,len(x_dense)):
    #     points.append(img2[int(y_dense[j]),int(x_dense[j])])
    # plt.imshow(img2)
    # plt.scatter(data[:,2], data[:,3], color='red', s=3, label='Original Pixels')
    # plt.plot(x_dense, y_dense, color='blue', label='Cubic Spline Fit')
    # plt.show()
    
    range_x = abs(x_dense - 512)
    x_value_mid = np.where(range_x ==range_x.min())[0][0]
    plt.plot(range_x, color='red', label='differene from image width')
    plt.plot([mean_vertebral_width]*512, color='green', label='Mean vertebral width')
    plt.plot([max_vertebral_width]*512, color='blue', label='Max vertebral width')
    plt.plot([min_vertebral_width]*512, color='orange', label='Min vertebral width')
    plt.plot([median_vertebral_width]*512, color='black', label='Median vertebral width')
    plt.legend()
    plt.scatter(x_value_mid,range_x.min())
    plt.annotate(f'{round(range_x.min())}', (x_value_mid, round(range_x.min(),2)), textcoords="offset points", xytext=(0,10), ha='center')
    plt.title(file_path.split('\\')[-1] +'_' +'Ratio of Difference to mean vertebral width = ' + str(round(range_x.min()/mean_vertebral_width,2)))
    plt.savefig(os.path.join(dst, 'file_analysis'+file_path.split('\\')[-1]))
    plt.close()
    ratios.append(round(range_x.min()/mean_vertebral_width,2))

plt.plot(ratios)
plt.ylim(0, 1.5)

plt.title('Ratio_region_available_to_right_wrt_vertebral_width')
plt.savefig(os.path.join(dst, 'overall_file_analysis.png'))
plt.close()

    # x_mid_1, y_mid_1, points_1 = spline_fit_check(n, m, img, img2, mean_vertebral_width,0.75, pts_detailed, img_name)
    
    # scale_x, scale_y = 512/w, 1024/h
    
    # resized_img = img.resize((512,1024))
    
    # gt_labels_scaled = gt_labels.copy()
    
    # gt_labels_scaled[:,0] = gt_labels[:,0] * scale_x
    # gt_labels_scaled[:,1] = gt_labels[:,1] * scale_y