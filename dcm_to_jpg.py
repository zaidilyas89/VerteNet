import os
import pydicom
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import tqdm
import argparse

# Create the parser
parser = argparse.ArgumentParser(description="DXA dicoms to jpgs conversion.")

# Add an argument for the folder path
parser.add_argument('--src_folder_path', 
                    type=str, nargs='?',  
                    default = './dcm1',  
                    help='Enter Source Folder.')
parser.add_argument('--dst_folder_path', 
                    type=str, nargs='?',  
                    default = './jpgs',  
                    help='Enter Destination Folder.')
parser.add_argument('--min_value', 
                    type=int, nargs='?',  
                    default = 0,  
                    help='Enter min value for normalization.')
parser.add_argument('--max_value', 
                    type=int, nargs='?',  
                    default = 4096,  
                    help='Enter max value for normalization.')
# Parse the arguments
args = parser.parse_args()

# Use the folder path argument
src_folder = args.src_folder_path
dst_folder = args.dst_folder_path
max_value = args.max_value
min_value = args.min_value

# Check if the folder exists
if not os.path.exists(dst_folder):
    # Create the folder if it doesn't exist
    os.makedirs(dst_folder)
    print(f"Folder created: {dst_folder}")
else:
    print(f"Folder already exists: {dst_folder}")


def dcm_to_jpg(dcm_file, dst_folder):
    # Load DICOM file
    dcm_data = pydicom.dcmread(dcm_file)

    # Get image data from DICOM
    image = dcm_data.pixel_array

    # Normalize pixel values to 0-255
    image = (image - min_value) / (max_value - min_value) * 255

    # Convert to unsigned 8-bit integer (uint8)
    image = image.astype(np.uint8)

    # Get the filename without extension
    file_name = os.path.join(dst_folder,os.path.splitext(os.path.basename(dcm_file))[0])

    # Save as PNG file with the same name
    png_file = file_name + ".jpg"
    plt.imsave(png_file, image, cmap='gray')

    print(f"Conversion completed. Saved as {png_file}.")


paths = glob(os.path.join(src_folder,'*.dcm'))
for path in tqdm.tqdm(paths):
    # name = path.split('\\')[-1].split('.dcm')[0] +'.jpg'
    dcm_to_jpg(path, dst_folder)
