# PyTorchUNET
 UNET using PyTorch for ML

The .py file 'realTimeInference' will inference videos in specified folder, if given segmentation model and classification model. 

The file hierarchy is as follows for inference:

    -   realTimeInference (runs)
        -   auxiliary (random functions to keep other files cleaner)
        -   model (load segmentation model)
        -   classifierNets (load classifier model)
        -   contourEllipseDetection (find regions in video and output list)
        -   newTracking (track id's from one frame to the next)

The file hierarchy is as follows for training:

    -   train
        -   auxiliary (random functions to keep other files cleaner)
        -   model(load predefined segmentation model)
        -   dataset (for loading)
    -   NEWevaluate (for testing segmentation model)



For post-processing, the parseCSV file handles the csv output, and will acquire all csv files in specified folder and concatenate into single csv file, provided that the shape of each csv file data structure is the same. 

The makeGIF function will make a GIF given an image source directory

The classifier file contains script to train the classification model. 

*** NOTE *** 
Both the segmentation and classification models were trained usign scripts from a different reposistory, for code and dataset cleanliness. However, the functions implemented are practically the same. 