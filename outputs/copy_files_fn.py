



from shutil import copy
from glob import glob
import os
from tqdm import tqdm

def copy_files(src_path = None, dst_path = None, data_type = 'png'):
    file_paths = glob(os.path.join(src_path,'*.'+ data_type))
    
    return