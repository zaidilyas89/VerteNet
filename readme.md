# VerteNet
A PyTorch implementation of VerteNet for vertebral landmarks localization in lateral view DXA images.
![Network Architecture](result/architecture.png)

## Dataset
Custom Dataset was used for the training of this model which has not been shared due to Data Ethics Limitations. 
sure the directory like this:
```                           
|-- data     
    |-- rain100L
        |-- train
            |-- rain
                norain-1.png
                ...
            `-- norain
                norain-1.png
                ...
        `-- test                                                        
    |-- rain100H
        same as rain100L
```
