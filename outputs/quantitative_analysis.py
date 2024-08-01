# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:50:21 2024

@author: zaidi
"""



import pandas as pd
import numpy as np
import os
import glob
from scipy.io import loadmat
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

def plot_landmarks(img, landmarks, save_path=None):
    # Open an image file
    # img = Image.open(image_path)
    
    # Convert PIL image to numpy array
    img_array = np.array(img)

    # Create a plot
    fig, ax = plt.subplots()
    ax.imshow(img_array)

    # Plot landmarks
    for (x, y) in landmarks:
        ax.plot(x, y, 'ro')  # 'ro' means red color, round shape

    # Display the plot
    plt.show()

    # Save the plot with landmarks if save_path is provided
    # if save_path:
    #     fig.savefig(save_path)

def normalized_mean_error(y_true, y_pred, normalization_value=None):
    """
    Calculate the Normalized Mean Error (NME).

    Parameters:
    y_true (array-like): True values.
    y_pred (array-like): Predicted values.
    normalization_value (float, optional): Value to normalize the mean error. If None, 
                                           the normalization value will be the mean of y_true.

    Returns:
    float: The normalized mean error.
    """
    # Convert input to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate the mean absolute error
    mean_error = np.median(np.abs(y_true - y_pred))
    
    # If no normalization value is provided, use the mean of the true values
    if normalization_value is None:
        normalization_value = np.mean(y_true)
    
    # Calculate the normalized mean error
    nme = mean_error / normalization_value
    
    return nme


gt_labels_path = r'D:\12_AAC_Semantic_Segmentation\Landmark Detection\dataPath\labels\val'
gt_img_path = r'D:\12_AAC_Semantic_Segmentation\Landmark Detection\dataPath\data\val'
guidenetv1_labels_path = r'D:\12_AAC_Semantic_Segmentation\Landmark Detection\outputs\GuideNetv1_predictions_detailed\test_detailed_outputs\landmarks'
guidenetv2_labels_path = r'D:\12_AAC_Semantic_Segmentation\Landmark Detection\outputs\GuideNetv3_predictions_detailed\test_detailed_outputs\landmarks'

file_paths_gt = glob.glob(os.path.join(gt_labels_path,'*.mat'))
file_paths_pred_v1 = glob.glob(os.path.join(guidenetv1_labels_path,'*.csv'))
file_paths_pred_v2 = glob.glob(os.path.join(guidenetv2_labels_path,'*.csv'))

data = {'nme_v1': [],
        'nme_v2': []}

for count, i in enumerate(tqdm(file_paths_gt)):
    fname = i.split('\\')[-1].split('.')[0]
    gt_labels = loadmat(i)['p2']
    img = Image.open(os.path.join(gt_img_path,fname+'.jpg'))
    w, h = img.size
    scale_x, scale_y = 512/w, 1024/h
    
    resized_img = img.resize((512,1024))
    
    gt_labels_scaled = gt_labels.copy()
    
    gt_labels_scaled[:,0] = gt_labels[:,0] * scale_x
    gt_labels_scaled[:,1] = gt_labels[:,1] * scale_y
    
    
    
    pred_labels_v1 = pd.read_csv(os.path.join(guidenetv1_labels_path,fname+'.csv'),index_col=False)
    x1 = pred_labels_v1.iloc[:,3:-1].values
    # x_v1 = np.concatenate((x1[:,0:2],x1[:,2:4],x1[:,4:6],x1[:,6:8]),axis=0)
    x_v1 = x1.reshape(24,2)

    pred_labels_v2 = pd.read_csv(os.path.join(guidenetv2_labels_path,fname+'.csv'),index_col=False)
    x2 = pred_labels_v2.iloc[:,3:-1].values
    # x_v2 = np.concatenate((x2[:,0:2],x2[:,2:4],x2[:,4:6],x2[:,6:8]),axis=0)
    x_v2 = x2.reshape(24,2)
    # plot_landmarks(resized_img, gt_labels_scaled, None)
    
    if count==0:
        x_gt_all = gt_labels_scaled.copy()
        x_v1_all = x_v1.copy()
        x_v2_all = x_v2.copy()
    else:
        x_gt_all = np.concatenate((x_gt_all,gt_labels_scaled),axis=0)
        x_v1_all = np.concatenate((x_v1_all,x_v1),axis=0)
        x_v2_all = np.concatenate((x_v2_all,x_v2),axis=0)
    
print(normalized_mean_error(x_gt_all, x_v1_all, normalization_value=1))
print(normalized_mean_error(x_gt_all, x_v2_all, normalization_value=1))


# print('Mean GuideNetv1 = '+ str(np.array(data['nme_v1']).mean()))
# print('Mean GuideNetv2 = '+ str(np.array(data['nme_v2']).mean()))

