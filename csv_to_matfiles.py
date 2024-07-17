


import pandas as pd
from tqdm import tqdm
from glob import glob
from scipy.io import loadmat
from scipy.io import savemat
import sys
import numpy as np
import os

csv_files = glob('./*.csv')
# data = loadmat('1009 - 140140692-c.jpg')['p2']



for i in range(0,len(csv_files)):

    file = csv_files[i]
    
    folder_name = file.split('_')[-1].split('.csv')[0]
    df = pd.read_csv(file)
    
    
    counts = df['fname'].value_counts()
    if (counts>24).sum() > 0:
        print('Error - Duplicates Found')
        sys.exit()
    
    if (counts<24).sum() > 0:
        print('Error - Missing Values Found')
        sys.exit()
    
    fnames = list(set(df['fname']))
    
    for j in range(0,len(fnames)):
        p2 = df[df['fname'] == fnames[j]][['x','y']].values.astype(np.uint16)
        if fnames[j].split('.')[-1] == 'png':
            name = fnames[j].split('.png')[0]+'.jpg'
        else:
            name = fnames[j]
        
        savemat(os.path.join(folder_name,name+'.mat'), {'p2': p2})

    
