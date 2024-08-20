# VerteNet
A PyTorch implementation of VerteNet for vertebral landmarks localization in lateral view DXA images.
![Network Architecture](result/architecture.png)

## Dataset
Custom dataset was used for the training of this model which has not been shared due to data ethics limitations. 
For training, testing, and prediction using VerteNet, make sure the directory like this:
```                           
|-- datapath     
    |-- data
        |--train
            trainimage1.jpg
            trainimage2.jpg
            ...
        |--test
            testimage1.jpg
            testimage2.jpg
            ...
        |--val
            valimage1.jpg
            valimage2.jpg
            ...        
    |--labels
        |--train
            trainimage1.jpg.mat
            trainimage2.jpg.mat
            ...
        |--val
            valimage1.jpg.mat
            valimage2.jpg.mat
            ...     
```
## Data Annotation
For data annotation, we used [makesense](https://github.com/SkalskiP/make-sense?tab=readme-ov-file) tools in the offline mode. While annotating images, keep the orientation of images as per the input image shown in the architecture image above. For the conversion of landmarks take the steps mentioned in [steps](annotations_conversion/Steps.txt).

## Train Model
To train the model, place the images and corresponding labels in the subdirectories as shown above and then run following command:
```
python main.py ----phase train
```

## Test Model
For the prediction on new images, first place the trained model's file as `./weights_spinal/model_last.pth`, and then place the images in .jpg format in the  `./datapath/data/test` folder and then run the following command:
```
python main.py --phase test 
```
The predictions would be saved in the folder`./detailed_outputs`in the following form:
```
|-- detailed_outputs
    |-- images_with_landmarks_and_IVGs
    |-- images_with_landmarks_only
    |-- images_with_offset_vectors
    |-- landmarks
    |-- original_images
```
## Abdominal Aorta Crop Detection
For the abdominal aorta crop detection, performs following steps:
**Step 1 (Optional - For CLSA dataset)**, use the following command to sort and rename DICOM files
```
python sorting_and_combining_and_naming.py --src_folder_path ./23SP004_F2 --dst_folder_path ./ 23SP004_F2 _dcm
```
**Step 2**: Use following command to convert DICOM files to jpg files
```
python dcm_to_jpg.py --src_folder_path ./dcm2 --dst_folder_path ./23SP004_F2_jpgs --min_value 0 --max_value 4096
```
**Note**: It is important to manually check few images of the DICOM files for the maximum and minimum value range for the pixels. If for example, you find a value of 3998 as a maximum value and a minimum value of 0, then you need to select min value =0, and max value =4096 (multiple of 2). DO NOT select as 3998 or 4000. It is important to manually check few images of the DICOM files for the maximum and minimum value range for the pixels. If for example, you find a value of 3998 as a maximum value and a minimum value of 0, then you need to select min value =0, and max value =4096 (multiple of 2). DO NOT select as 3998 or 4000.

**Step 3**: Copy the jpg files to ```./VerteNet/dataPath/data/test```

**Step 4**: Paste the trained model file at the location ```./VerteNet/weights_spinal/model_last.pth.```

**Step 5**: Run the following command to get predictions from VerteNet: 

```
python main.py --phase test --dxa_dataset clsa
```

You will get a folder structure like this 

```
|-- datasetname_detailed_outputs
    |-- images_with_landmarks_and_IVGs
    |-- images_with_landmarks_only
    |-- images_with_offset_vectors
    |-- landmarks
    |-- original_images
```

**Step 6** Go to ```./VerteNet/dataPath/outputs``` and run the following command:
```
python Algorithm_Aorta_Clip_Detection.py  --src_folder_path VerteNet_predictions/clsa_f2_5000_detailed_outputs --dst_folder_path clsa --factors_range_min 0.9 --factors_range_max 1.2 --dpi 150
```
A new folder will be generated with following structure:
```
|-- datasetname
    |--imgs
    |--landmarks
    |--outputs
    |--splines 
```
The ‘splines’ folder contains the resultant files with the format ```Aorta_Clip_Flag_flagstatus_filename.jpg```
