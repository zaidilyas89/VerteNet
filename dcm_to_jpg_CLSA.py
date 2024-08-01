import os
import pydicom
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
import tqdm

src_folder = './imgs/'
dst_folder = './jpgs/'

def dcm_to_png(dcm_file, dst_folder):
    # Load DICOM file
    dcm_data = pydicom.dcmread(dcm_file)

    # Get image data from DICOM
    image = dcm_data.pixel_array

    # Normalize pixel values to 0-255
    image = (image - 0) / (4095 - 0) * 255

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
    dcm_to_png(path, dst_folder)
