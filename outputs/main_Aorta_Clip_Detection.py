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
import aorta_clip_detection, aorta_clip_detection_3
from aorta_clip_detection_3 import sort_and_copy
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

data_folder = './GuideNetv3_predictions_detailed/combined_us_model_mros'
landmarks_folder = './landmarks'

import pandas as pd
from shutil import copy
from custom_utilities import *

# fctr = 0
for fctr in tqdm(list(np.arange(0.5, 1.6, 0.1))):
    fctr = round(fctr,1)
    dataset = {}
    sub_folders = list(os.walk(data_folder))[0][1]
    
    data_folders = './us_model'
    data_folders_sub_folders = ['imgs', 'landmarks','outputs','splines']
    detailed_IVGs_folder = os.path.join(data_folders,data_folders_sub_folders[-2])
    create_folders_with_subfolders(data_folders, data_folders_sub_folders)
    spline_folder = os.path.join(data_folders,data_folders_sub_folders[-1])
    factor_folder = os.path.join(spline_folder,'Factor_'+str(fctr))
    create_folder(os.path.join(spline_folder,'Factor_'+str(fctr)))
    # create_folders_with_subfolders(factor_folder, sub_folders_acc)
    src_folder = './GuideNetv3_predictions_detailed/combined_us_model_mros/original_images'
    dst_folder = os.path.join(data_folders,'imgs')
    create_folder(dst_folder)
    # folder_with_fnames = './imgs_70/without_clip'
    # file_paths = glob(os.path.join(folder_with_fnames,'*.png'))
    # fnames = []
    # for i in file_paths:
    #     fnames.append(i.split('\\')[-1])
    
    # fnames_based_copy_files(src_folder, dst_folder, fnames)
    # src_folder = './imgs_70/imgs_with_clipping_as_per_John'
    # simple_copy_files(src_folder, dst_folder)
    
    
    
    
    
    # without_clip_gt_folder = './imgs_70/without_clip'
    # clip_gt_folder = './imgs_70/imgs_with_clipping_as_per_John'
    # file_paths_woc = glob(os.path.join(without_clip_gt_folder,'*.png'))
    # fnames_woc = []
    # for i in file_paths_woc:
    #     fnames_woc.append(i.split('\\')[-1])
    # file_paths_wc = glob(os.path.join(clip_gt_folder,'*.png'))
    # fnames_wc = []
    # for i in file_paths_wc:
    #     fnames_wc.append(i.split('\\')[-1])
    # fnames_combined = []
    # fnames_combined = [*fnames_woc,*fnames_wc]
    
    file_paths_wc = glob(os.path.join(src_folder,'*.png'))
    fnames_combined = []
    for i in file_paths_wc:
        fnames_combined.append(i.split('\\')[-1])
    data_border_clip=[]
    data_region_clip=[]
    for idx, img_name in enumerate(tqdm(fnames_combined)):
        # if img_name not in imgs_analysis:
        #     continue
        # else:
        #     # path = r'D:\12_AAC_Semantic_Segmentation\Landmark Detection\outputs\imgs'
        #     # copy(os.path.join(imgs_path,img_name),os.path.join(path,img_name))
        #     # continue
            # if img_name != 'de_SBH-09072015_090913.png':
            #     continue
        
            
            landmark_name = img_name.split('.png')[0]+'.csv'
            
            img = Image.open(os.path.join(data_folder, 'original_images',img_name))
            img = np.array(img)
            orig_img = img.copy()
            pts = pd.read_csv(os.path.join(data_folder, 'landmarks',landmark_name),index_col=False)
            pts0 = pts.iloc[:,1:].values
            # pts_ = pts0.iloc[:,3:-1].values
            # pts0 = x1.reshape(24,2)
            
            ori_image_regress, ori_image_points, ori_image_points_points_only, pts_detailed = draw_points_detailed.draw_landmarks_regress_test(pts0,
                                                                                                   img.copy(),
                                                                                                   img)
            
            # plt.tight_layout()
        
            # plt.imshow(ori_image_points)
            # plt.savefig(os.path.join(detailed_IVGs_folder,img_name), bbox_inches='tight')
            # plt.close()
            # print('\n' + img_name +'____' +str(idx))
            border_clip_flag, region_clip_flag = aorta_clip_detection_3.aorta_clip_detection(orig_img, ori_image_points, pts_detailed, img_name, FACTOR = fctr, check_region_clipping = True, spline_save_folder = factor_folder)
            if border_clip_flag == 1:
                data_border_clip.append(img_name)
            if region_clip_flag == 1:
                data_region_clip.append(img_name)
            
            # fname_path = glob(os.path.join(factor_folder,'*'+name+'*'))[0]
            # fname = fname_path.split('\\')[-1]
            # dst = os.path.join(factor_folder,fname)
            # copy(fname_path,dst)            
        
    # dataset[sub_folder] = {'region_clip':data_region_clip,
    #                        'border_clip':data_border_clip}
    
    # data1 = {'name': data_border_clip,
    #          'class': [1]*len(data_border_clip)}
    # df_wc = pd.DataFrame(data1)
    # universal_set = set(fnames_combined)
    # clip_set = set(data_border_clip)
    # allowed_set = universal_set - clip_set
    # allowed = list(allowed_set)
    # data2 = {'name': allowed,
    #          'class': [0]*len(allowed)}
    # df_woc = pd.DataFrame(data2)
    
    
    # data_gt_wc = {'name': fnames_wc,
    #               'class': [1]*len(fnames_wc)}
    # df_gt_wc = pd.DataFrame(data_gt_wc)
    
    # data_gt_woc = {'name': fnames_woc,
    #               'class': [0]*len(fnames_woc)}
    # df_gt_woc = pd.DataFrame(data_gt_woc)
    
    # df_gt = pd.concat([df_gt_wc,df_gt_woc],axis = 0)
    # df_gt_sorted = df_gt.sort_values(by='name')
    # df_gt_sorted.reset_index(drop=True, inplace=True)
    
    
    # df_pred = pd.concat([df_wc,df_woc],axis = 0)
    # df_pred_sorted = df_pred.sort_values(by='name')
    # df_pred_sorted.reset_index(drop=True, inplace=True)
    
    # preds = list(df_pred_sorted['class'])
    # labels = list(df_gt_sorted['class'])
    
    # cnf_matrix = confusion_matrix(preds, labels,labels=[0,1])
    # # https://towardsdatascience.com/multi-class-classification-extracting-performance-metrics-from-the-confusion-matrix-b379b427a872
    # FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix) 
    # FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    # TP = np.diag(cnf_matrix)
    # TN = cnf_matrix.sum() - (FP + FN + TP)
    # FP = FP.astype(float)
    # FN = FN.astype(float)
    # TP = TP.astype(float)
    # TN = TN.astype(float)
    
    # # Sensitivity, hit rate, recall, or true positive rate
    # TPR = TP/(TP+ 0.000000000000000001+FN + 0.000000000000000001)
    # # Specificity or true negative rate
    # TNR = TN/(TN+ 0.000000000000000001+FP + 0.000000000000000001) 
    
    
    # # Precision or positive predictive value
    # PPV = TP/(TP+ 0.000000000000000001+FP+ 0.000000000000000001)
    # # Negative predictive value
    # NPV = TN/(TN+ 0.000000000000000001+FN+ 0.000000000000000001)
    
    # # # Fall out or false positive rate
    # # FPR = FP/(FP+TN)
    # # # False negative rate
    # # FNR = FN/(TP+FN)
    # # # False discovery rate
    # # FDR = FP/(TP+FP)
    # # # Overall accuracy for each class
    # ACC = (TP+TN)/(TP+FP+FN+TN)
    # data_cm = {'FP':FP,
    #            'FN':FN,
    #            'TP':TP,
    #            'TN':TN,
    #            'TPR':TPR,
    #            'TNR':TNR,
    #            'PPV':PPV,
    #            'NPV':NPV,
    #            'ACC':ACC}
    
    # df_cm = pd.DataFrame(data_cm)
    # df_cm.to_csv(os.path.join(spline_folder,'Factor_'+str(fctr),str(fctr)+'_Confusion_Matrix.csv'), index=False)
    # # Create a confusion matrix display object
    # cmd = ConfusionMatrixDisplay(confusion_matrix=cnf_matrix, display_labels=[0,1])
    
    # # fig, ax = plt.subplots()
    # # cmd.plot(cmap='Blues', ax=ax)
    
    # # Plot the confusion matrix
    # cmd.plot(cmap=plt.cm.Blues)
    # plt.title('Confusion Matrix Aorta Clip Detection')
    # plt.xlabel('Predicted Label')
    # plt.ylabel('True Label')
    
    
    # # Save the figure
    # plt.savefig(os.path.join(spline_folder,'Factor_'+str(fctr),str(fctr)+'_confusion_matrix.png'))
    
    # # Show the plot
    # plt.show()
    # plt.close()        
    # # sort_and_copy(df_labels_sorted, df_pred_sorted)
    # aa = {'name': list(df_gt_sorted['name'].values),
    #       'preds': list(df_pred_sorted['class'].values),
    #       'gts': list(df_gt_sorted['class'].values)}
    # a = pd.DataFrame(aa)                        
    # for name in a['name']:
    #     if (a[a['name'] == name]['preds'] == 0).all() and (a[a['name'] == name]['gts'] == 0).all():
    #         fname_path = glob(os.path.join(factor_folder,'*'+name+'*'))[0]
    #         fname = fname_path.split('\\')[-1]
    #         dst = os.path.join(factor_folder,'TN',fname)
    #         copy(fname_path,dst)
    #     elif (a[a['name'] == name]['preds'] == 0).all() and (a[a['name'] == name]['gts'] == 1).all():
    #         fname_path = glob(os.path.join(factor_folder,'*'+name+'*'))[0]
    #         fname = fname_path.split('\\')[-1]
    #         dst = os.path.join(factor_folder,'FN',fname)
    #         copy(fname_path,dst)
    #     elif (a[a['name'] == name]['preds'] == 1).all() and (a[a['name'] == name]['gts'] == 0).all():
    #         fname_path = glob(os.path.join(factor_folder,'*'+name+'*'))[0]
    #         fname = fname_path.split('\\')[-1]
    #         dst = os.path.join(factor_folder,'FP',fname)
    #         copy(fname_path,dst)
    #     elif (a[a['name'] == name]['preds'] == 1).all() and (a[a['name'] == name]['gts'] == 1).all():
        # fname_path = glob(os.path.join(factor_folder,'*'+name+'*'))[0]
        # fname = fname_path.split('\\')[-1]
        # dst = os.path.join(factor_folder,'TP',fname)
        # copy(fname_path,dst)            
