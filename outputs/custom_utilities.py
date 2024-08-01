



from shutil import copy
from glob import glob
import os
from tqdm import tqdm

def create_folder(path):
    # Check if the folder already exists
    if not os.path.exists(path):
        # If not, create the folder
        os.makedirs(path)
        print(f"Created folder: {path}")
    else:
        print(f"Folder already exists: {path}")

def create_folders_with_subfolders(root_directory, subfolders):


    # Create the root directory
    create_folder(root_directory)

    # Create each subfolder within the root directory
    for subfolder in subfolders:
        subfolder_path = os.path.join(root_directory, subfolder)
        create_folder(subfolder_path)

    # Check and print the existence of each subfolder
    for subfolder in subfolders:
        subfolder_path = os.path.join(root_directory, subfolder)
        if os.path.exists(subfolder_path):
            print(f"Confirmed existence: {subfolder_path}")
        else:
            print(f"Subfolder does not exist and was not created: {subfolder_path}")

# # Example usage
# root_directory = "main_folder"
# subfolders = ["subfolder1", "subfolder2", "subfolder3"]

# create_folders_with_subfolders(root_directory, subfolders)



def simple_copy_files(src_path = None, dst_path = None, data_type = 'png'):
    
    'Please provide linux based paths e.g. ./folder/subfolder/data_type'
    
    
    
    failed_copy_fnames = []
    fnames = []
    flag = 0
    
    create_folder(dst_path)
    
    file_paths = glob(os.path.join(src_path, '*.'+ data_type))
    for src_file_path in tqdm(file_paths):
        fname = src_file_path.split('\\')[-1]
        fnames.append(fname)
        dst_file_path = os.path.join(dst_path, fname)
        try:
            copy(src_file_path,dst_file_path)
            print('Copy Successful')
        except:
            failed_copy_fnames.append(fname)
            print('Copy Failed!')
            
    if len(failed_copy_fnames)>0:
        print('Some Files didn''t copy!')
        flag = 1
    else:
        print('All files copied successfully!')
        flag = 0
    return flag, fnames, failed_copy_fnames

# def files_check_in_folder(fnames = None, folder = None, data_type = '.png'):
#     flag = 0
#     flag = 1
#     file_paths = glob(os.path.join(folder, '*.'+ data_type))
    # fnames_ = []
    # for src_file_path in tqdm(file_paths):
    #     fname = src_file_path.split('/')[-1]
    #     fnames_.append(fname)
    
#         # List of names to check
#     names_to_check = ["Alice", "Bob", "Charlie"]
    
#     # List where to check the names
#     existing_names = ["Alice", "David", "Bob", "Eve"]
    
#     # Check if names in names_to_check exist in existing_names
#     found_names = [name for name in names_to_check if name in existing_names]
    
#     # Output the found names
#     print("Names found:", found_names)

#     return flag

def fnames_based_copy_files(src_path = None, dst_path = None, fnames_for_copy = None, data_type = 'png'):
    
    'Please provide linux based paths e.g. ./folder/subfolder/data_type'
    failed_copy_fnames = []
    fnames = []
    flag = 0
    
  
    
    create_folder(dst_path)
    
    for fname_to_copy in tqdm(fnames_for_copy):
        src_file_path = os.path.join(src_path, fname_to_copy)
        dst_file_path = os.path.join(dst_path, fname_to_copy)
        try:
            copy(src_file_path,dst_file_path)
            print('Copy Successful')
        except:
            failed_copy_fnames.append(fname_to_copy)
            print('Copy Failed!')
            
        if len(failed_copy_fnames)>0:
            print('Some Files didn''t copy!')
            flag = 1
        else:
            print('All files copied successfully!')
            flag = 0
    return flag, fnames, failed_copy_fnames


