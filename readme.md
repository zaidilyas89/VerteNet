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
