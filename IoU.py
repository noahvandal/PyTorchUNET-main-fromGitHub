from auxiliary import determineROI, whatPercentIsClass
import os
import random
import numpy as np


doISaveImage = True

rootPath = '/home/noahvandal/my_project_dir/my_project_env/UNET_Color/UNET_MC_PyTorch/FineTuneMarchModel/'

testIdentifier = 'IoUTest_PureTrainHPNEBias'

imgDir = rootPath + 'PureFineTuneVal/Masks/'
maskDir = rootPath + 'PureFineTuneVal/TestHPNEbias/'

csvSave = rootPath + testIdentifier + '.csv'

imgList = os.listdir(imgDir)


roiData = []
percentData = []

isConcat = True ## if the mask output has input side by side. 

for i, img in enumerate(imgList):
    imgPath = imgDir + img
    maskPath = maskDir + img

    imgName = img[:-4]

    ClassIou, totalIou = determineROI(imgPath, maskPath, isConcat, isPSBead=False)

    percent = whatPercentIsClass(imgPath)
    percentData.append([percent[0], percent[1], percent[2], percent[3]])

    percentAvg = np.array(percentData)
    percentAvg = np.average(percentAvg, axis=0)

    roiData.append([imgName, ClassIou[0], ClassIou[1],
                   ClassIou[2],totalIou])

    if doISaveImage:
        roiarray = np.array(roiData)
        np.savetxt(csvSave, roiarray, fmt="%s", delimiter=',',
                   header='ImgName,HPNE IoU,MIA IoU,Background IoU,Total IoU')

    if i % 10 == 0:
        print(i)
