# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 13:20:17 2021

@author: zaidi
"""

import numpy as np
import cv2
import os
from tqdm import tqdm
import random
import math
from glob import glob
import scipy.io

#-----------------------------------------------------------------------------
SRC_FOLDER = os.path.join(os.getcwd(),'data_backup')
TRAIN_IMG_FOLDER = os.path.join(os.getcwd(),'dataPath','data','train')
TEST_IMG_FOLDER = os.path.join(os.getcwd(),'dataPath','data','test')
VALID_IMG_FOLDER = os.path.join(os.getcwd(),'dataPath','data','val')


TRAIN_LABEL_FOLDER = os.path.join(os.getcwd(),'dataPath','labels','train')
TEST_LABEL_FOLDER = os.path.join(os.getcwd(),'dataPath','labels','test')
VALID_LABEL_FOLDER = os.path.join(os.getcwd(),'dataPath','labels','val')
#-----------------------------------------------------------------------------
f = glob(SRC_FOLDER+"/*jpg")
fnames = []
for i in f:
    fnames.append(os.path.basename(i))

SEED = 448
random.seed(SEED)
random.shuffle(fnames)
random.shuffle(fnames)

VALID_PERCENTAGE = 0.15
TEST_PERCENTAGE = 0.15

TEST_V = math.ceil(TEST_PERCENTAGE * len(fnames))
VALID_V = math.ceil(VALID_PERCENTAGE* len(fnames))
TRAIN_V = len(fnames)-VALID_V-TEST_V

img_fname_test = fnames[-TEST_V:]
img_fname_valid = fnames[-VALID_V-TEST_V:-TEST_V]
img_fname_train = fnames[0:-VALID_V-TEST_V]
#-----------------------------------------------------------------------------
def transfer_images(SRC,DST,filenames):
    
    
    for i in filenames:
        a = cv2.imread(os.path.join(SRC,i),-1)
        cv2.imwrite(os.path.join(DST,i),a)
        
    return

def transfer_labels(SRC,DST,filenames):
    
    
    for i in filenames:
        mat = scipy.io.loadmat(os.path.join(SRC,i+'.mat'))
        scipy.io.savemat(os.path.join(DST,i+'.mat'),mat)
        
    return mat
#-----------------------------------------------------------------------------

transfer_images(SRC_FOLDER,TRAIN_IMG_FOLDER,img_fname_train)
transfer_images(SRC_FOLDER,TEST_IMG_FOLDER,img_fname_test)
transfer_images(SRC_FOLDER,VALID_IMG_FOLDER,img_fname_valid)

transfer_labels(SRC_FOLDER,TRAIN_LABEL_FOLDER,img_fname_train)
transfer_labels(SRC_FOLDER,TEST_LABEL_FOLDER,img_fname_test)
transfer_labels(SRC_FOLDER,VALID_LABEL_FOLDER,img_fname_valid)
