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
For data annotation, we used [makesense](https://github.com/SkalskiP/make-sense?tab=readme-ov-file) tools in the offline mode. Link to download the offline version: 

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

