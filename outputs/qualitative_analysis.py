# -*- coding: utf-8 -*-
"""
Created on Mon May 20 09:36:52 2024

@author: zaidi
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import glob
from PIL import Image
import matplotlib.pyplot as plt

folders = ['GuideNetv1_predictions_detailed','GuideNetv2_predictions_detailed','GuideNetv3_predictions_detailed','qualitative_analysis']
datasets = list(os.walk(folders[0]))[0][1]
data = list(os.walk(os.path.join(folders[0], datasets[0])))[0][1]



for i in range(3,len(datasets)):
    dataset = datasets[i]
    
    
    os.makedirs(os.path.join(folders[2],dataset), exist_ok=True)
     
    image_paths = glob.glob(os.path.join(folders[0],dataset,data[-1],'*.png'))
    
    
    
    for image_path in tqdm(image_paths):
        image_name = image_path.split('\\')[-1]
        guidenetv1_image_path = os.path.join(folders[0],datasets[i],data[0],image_name)
        guidenetv2_image_path = os.path.join(folders[1],datasets[i],data[0],image_name)
        guidenetv3_image_path = os.path.join(folders[2],datasets[i],data[0],image_name)
        orig_img = Image.open(image_path)
        guidenetv1_img = Image.open(guidenetv1_image_path)
        guidenetv2_img = Image.open(guidenetv2_image_path)
        guidenetv3_img = Image.open(guidenetv3_image_path)
        # plt.imshow(orig_img)
        # Create a figure with a specified size
        fig, axes = plt.subplots(1, 4, figsize=(15, 15))  # Adjust the figsize as needed
        
        # Display each image in its respective subplot
        axes[0].imshow(orig_img)
        axes[0].axis('off')  # Hide the axis
        axes[0].set_title('input image')
        
        axes[1].imshow(guidenetv1_img)
        axes[1].axis('off')  # Hide the axis
        axes[1].set_title('GuideNetv1 Image')
        
        axes[2].imshow(guidenetv2_img)
        axes[2].axis('off')  # Hide the axis
        axes[2].set_title('GuideNetv2 Image')    
        # Display the plot
                
        axes[3].imshow(guidenetv3_img)
        axes[3].axis('off')  # Hide the axis
        axes[3].set_title('GuideNetv2_Improved Image')  
        plt.tight_layout()
    
        # plt.show()
        plt.savefig(os.path.join(folders[3],dataset,image_name), bbox_inches='tight')
        plt.close()
