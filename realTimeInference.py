from contourEllipseDetection import ellipseDetection, processOneHotIntoClassBW, getEllipsesFromClassList, getEllipsesFromClassListClassifier
# from PYtracking import Tracking, postVideoProcessList
# from PYtracking import postVideoProcessList
import torch
import cv2
import csv
from sys import stdout
import numpy as np
from model import UNET
from IPython.display import clear_output
from auxiliary import saveImageOutput, putEllipsesOnImage, suppressUndesirableEllipses, outputRegions, saveImage
from dataset import cellColor2Label, color2label, outputClassImages, RGBtoBW
from evaluate import onehot_to_rgb
from torchvision import transforms
from newTracking import Tracking, postVideoProcessList
import time
import os
import matplotlib.pyplot as plt
from classifier import Classify
from classifierNets import ClassifierHyperparam_v2

if torch.cuda.is_available():
    DEVICE = 'cuda:1'
else:
    DEVICE = 'cpu'


IMAGE_SHAPE = (960, 1280)


def onehotToBW(image,outputAsRGB=False):   
    # if isTensor:
        # print(image.shape)
        # image = convertTensorTypeToNumpy(image)
        # print(image.shape)

    image = np.array(image)
    image = image[0,:,:,:] ## getting rid of batch dimension
    image = np.transpose(image, (1, 2, 0))  ## flipping for cv2 sake
    output = np.zeros([image.shape[0], image.shape[1]])
    output = np.expand_dims(output, axis=-1)
    image = np.argmax(image, axis=-1, keepdims=True)
    # print(image.shape)
    output[np.all(image == 1, axis=-1)] = 255 ## background 
    output[np.all(image == 0, axis=-1)] = 0 ## foreground
    output = output.astype('uint8')
    # print(np.sum(np.all(output == [255], axis=-1)))
    # print(output.shape)
    # showImage(output)

    if outputAsRGB:
        output = cv2.cvtColor(output, cv2.COLOR_GRAY2RGB)

    # showImage(output)
    return output

def main(videoPath, saveImagePath, saveCSVPath, saveAccPath, modelPath, classifyModelPath, magnification):

    headerFile = ['ID', 'Coordinates', 'Axes Length', 'Number of Frames Present',
                  'Average Speed (px)', 'Identity', 'Number of frames as identity', 'Diameter', 'Start Frame', 'End Frame', 'HPNE Class Prob', 'MIA Class Prob']  # data for csv output

    model = UNET(in_channels=3, classes=2)
    checkpoint = torch.load(modelPath, map_location=torch.device(DEVICE))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f'{modelPath} has been loaded and initialized')

    model = model.to(DEVICE)


    """
    The following section of code is for the classification network, if present. 
    """
    # classificationModel = Classify(1) ## batch size of 1
    classificationModel = ClassifierHyperparam_v2()
    checkpoint = torch.load(classifyModelPath)
    classificationModel.load_state_dict(checkpoint['model_state_dict'])


    # model = torch.load(modelPath).to(DEVICE)

    GlobalList = []
    CurrentDict = {}  # list of all saved points that were inferenced in frames

    id = 0

    nameList = ['HPNE', 'MIA', 'PSBead', 'All']

    classifyList = ['HPNE','MIA']

    # magnification = 25

    pixelChange = 1  # just tracking movement of pixels through frames
    changeinpixel = 0
    magnification = 25  # should be set for entire frame

    # loop for each frame
    iterator = 1  # count how many frames pass

    src = cv2.VideoCapture(videoPath)

    ##list to keep accuracy values
    hpneMiaAcc = []
    while (src.isOpened()):  # qualifier like 'video open' or something here.

        startFrame = time.time()

        ret, frame = src.read()

        if frame is None:
            break

        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)  ## ensuring image is of grayscale type, but with 3 color channels

        # origImg = np.resize(frame,[960, 1280], interpolation=cv2.INTER_CUBIC)
        # print(frame.shape)
        origImg = cv2.resize(frame, [IMAGE_SHAPE[1], IMAGE_SHAPE[0]],
                             interpolation=cv2.INTER_LINEAR)  # have to flip dimensions due to being cv2 ** uggghhh***

        origImgGray = cv2.cvtColor(origImg, cv2.COLOR_RGB2GRAY)
        origImgGray = cv2.cvtColor(origImgGray, cv2.COLOR_GRAY2RGB) ## convert to gray and back, losing color data.


        totalEllipses = []

        # frame = preProcess(frame)  # pre-process image
        # frame = frame.preProcessImage()
        # print(origImg.shape)
        # frame = np.transpose(frame, [2, 0, 1])
        frame = cv2.resize(
            frame,  [IMAGE_SHAPE[1], IMAGE_SHAPE[0]], interpolation=cv2.INTER_LINEAR)
        frame = np.transpose(frame, [2, 0, 1])
        frame = np.reshape(frame, (1,)+frame.shape)
        # frame = np.resize(frame, (960, 1280))
        frame = frame / 255
        frame = torch.from_numpy(frame)
        frame = torch.tensor(frame, dtype=torch.float)
        # print(frame)
        # frame = transforms.ToTensor()(frame)

        # print('tensorshape', frame.shape)

        with torch.no_grad():
            frame = frame.to(DEVICE)
            inferenceStart = time.time()
            output = model(frame)  # output from trained unet model
            inferenceEnd = time.time()

        # convert image to each class type
        # getting b/w mask for each class present
        # print(type(output))


        output = output.detach().cpu().numpy()  ## to pull from gpu for inferencing on cpu

        
        '''
        Functions to use if doing unet classification:: 
        
        allimgs = outputClassImages(output, cellColor2Label)  ## getting a b/w mask for each class type present.

        # getting ellipses for each class present
        # totalEllipses = getEllipsesFromClassList(allimgs, nameList)

        '''

        # getting actual color mapping of all classes present
        # totalOutputImage = onehot_to_rgb(output, cellColor2Label)
        # bwImage = RGBtoBW(totalOutputImage, False) ## outputs a b/w mask from classwise segmentation. Easier than updating network and weights each time class switches. 
        # bwImage = np.bitwise_not(bwImage)
        bwImage = onehotToBW(output)
        bwImage = np.bitwise_not(bwImage)
        # plt.imshow(origImgGray, interpolation='nearest')
        # plt.show()
        # plt.imshow(totalOutputImage, interpolation='nearest')
        # plt.show()
        # plt.imshow(bwImage, interpolation='nearest')
        # plt.show()

        # getting ellipses for each class present
        totalEllipses, imageList = getEllipsesFromClassListClassifier(bwImage,origImgGray,classificationModel, classifyList, iterator)

        
        # print(totalEllipses)
        for i, img in enumerate(imageList):
            imName = str(iterator) + '_' + str(i) + '.png'
            saveImage(img, imName, saveImagePath)
        # placeholdellipse = totalEllipses.copy()
        # print(pixelChange)
        # print(len(totalEllipses))

        tracker = Tracking(CurrentDict, GlobalList, totalEllipses,
                           id, changeinpixel, pixelChange, magnification)   ## update list based on points in previous frame. 

        CurrentDict, id, GlobalList, _, changeinpixel = tracker.comparePointsList()

        for ellipse in CurrentDict.values():
            acc = ellipse[11]
            if len(acc) == 4:
                # print(ellipse)
                acc = ellipse[11]
                # print(acc)
                hpneAcc = acc[2]
                miaAcc = acc[3]
                hpneMiaAcc.append([hpneAcc, miaAcc])

        ## getting ellipses from image
        ellipseList = []
        for i, ellipse in enumerate(CurrentDict.values()):
            ell = ellipse
            ell = [ell[0], ell[1], ell[2], ell[3]]  ## only outputting x,y,w,h
            ellipseList.append(ell)
        

        ## making frame right dimensions for output
        segmentImg = frame.detach().cpu().numpy()
        segmentImg = np.array(segmentImg)
        segmentImg = segmentImg[0,:,:,:]
        segmentImg = np.transpose(segmentImg, [1, 2, 0])        
        # _ = outputRegions(segmentImg, str(iterator), ellipseList, saveImagePath)

        # GlobalList.extend(outOfFrameList)
        # print(changeinpixel)
        # print('global list', GlobalList)
        # print('pixel change', changeinpixel)

        # calculating change in pixel value from frame to frame; can determine flow rate from this.
        pixelChange = (pixelChange*iterator + changeinpixel) / iterator

        # print(CurrentDict)
        bwImage = np.bitwise_not(bwImage)
        bwImage = cv2.cvtColor(bwImage, cv2.COLOR_GRAY2RGB)
        imgs = putEllipsesOnImage(bwImage, CurrentDict, magnification)
        # print(origImg.shape, imgs.shape)
        saveImageOutput(origImg, imgs, str(iterator) + '_All',
                        saveImagePath, doISave=True, showOutput=False)  # ensure save is set to true to save output

        # print('Global List Before', GlobalList)
        # listcompare = postVideoProcessList(GlobalList)
        # GlobalList = listcompare.SingleComparePointsGlobal()
        # GlobalList = removeSimilar(GlobalList)

        clear_output(wait=True)
        # print(len(totalEllipses))
        # for entry in CurrentDict:
        #     print(entry, CurrentDict[entry])
        # for entry in GlobalList:
            # print(entry)
        listcompare = postVideoProcessList(GlobalList)
        GlobalList = listcompare.SingleComparePointsGlobal()
        # print('Global List')
        # for entry in GlobalList:
            # print(entry)
        # print('Global List', GlobalList)

        endFrame = time.time()

        # if iterator == 133:  ## incase i want to stop at a certain frame
            # break

        # clear_output(wait=True)
        stdout.flush()
        print('Frame Number: ', iterator)
        print('Frame Time: ', endFrame-startFrame)
        print('Inference Time: ', inferenceEnd - inferenceStart)

        iterator += 1

        # clear_output(wait=True)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
            # break

        with open(saveCSVPath, 'w', newline='') as csvwriter:
            write = csv.writer(csvwriter)
            write.writerow(headerFile)
            write.writerows(GlobalList)

        with open(saveAccPath, 'w', newline='') as csvwriter:
            write = csv.writer(csvwriter)
            # write.writerow(headerFile)
            write.writerows(hpneMiaAcc)

        # clear_output(wait=True)
        # print(CurrentDict)
    with open(saveCSVPath, 'w', newline='') as csvwriter:
        write = csv.writer(csvwriter)
        write.writerow(headerFile)
        write.writerows(GlobalList)
    

    src.release()
    # cv2.destroyAllWindows()

    with open(saveCSVPath, 'w', newline='') as csvwriter:
        write = csv.writer(csvwriter)
        write.writerows(GlobalList)



def runMultipleVideos(rootPath, extraName,dateFolder, modelPath, classifyModelPath):
    dirlist = os.listdir(rootPath)
    videoList = []

    for file in dirlist:
        if file[-4:] == '.mp4': ## ensuring only videos get inferences
            videoList.append(file)

    for i, video in enumerate(videoList):
        videoName = video[:-4]

        saveImagePath = rootPath + 'Output/' + dateFolder + str(videoName) + extraName + '/'
        doesPathExist = os.path.exists(saveImagePath)

        if doesPathExist == False:
            os.mkdir(saveImagePath)
        
        saveCSVPath = rootPath + 'Output/' + dateFolder + str(videoName) + extraName + '.csv'
        saveAccPath = rootPath + 'Output/' + dateFolder + str(videoName) + extraName + '_acc.csv'

        videoPath = rootPath + video

        print(videoPath)

        magnification = 25

        # if i == 0:
        main(videoPath, saveImagePath, saveCSVPath,saveAccPath, modelPath,classifyModelPath, magnification)



if __name__ == '__main__':

    rootPath = '/home/noahvandal/my_project_dir/my_project_env/UNET_Color/'

    # saveImagePath = rootPath + 'DatasetFeb10/OutputImages/HPNE_230203121917/'
    # saveCSVPath = rootPath + 'DatasetFeb10/HPNE_230203121917.csv'

    # define model path here
    # modelPath = rootPath + \
    # 'Videos/Train Images/3D Train/FullTrainDataset/Weights/01182023_ReduceLRonPlateau_model_2xscale.pt'

    # modelPath = rootPath + 'UNET_MC_PyTorch/FineTuneModels/021123_2x_3c_Train_model.pt'

    # modelPath = rootPath + 'UNET_MC_PyTorch/FineTuneMarchModel/030723_2x_3c_PureTrainHPNEbias_v7_model.pt'
    modelPath = '/home/noahvandal/my_project_dir/my_project_env/TrainUNET/Models/041123_2c_v3_wAugs_p10.pt'

    classifyModelPath = rootPath + 'HybridNet/Dataset/Model/051023_2c_Hyperparam_v2_fillTrain_noAug_drop10_bn_binaryBack_adamW_631split_earlyTerminate_quintMia.pt'
    # videoPath = rootPath + 'DatasetFeb10/HPNE/230203121917.mp4'
    # videoPath = rootPath + \
    # 'Videos/August 2022/20um/5ul_min/25x mag/1280x960 px/5_25_1280_0.mp4'
    # magnification = 25
    name = '051023_2c_Hyperparam_v2_fillTrain_noAug_drop10_bn_binaryBack_adamW_631split_earlyTerminate_quintMia'

    dateFolder = '11_May/'  ## the name of folder within output folder

    # main(videoPath, saveImagePath, saveCSVPath,
        #  modelPath, magnification=magnification)
    # runMultipleVideos(rootPath + 'DatasetFeb10/HPNE/',date, modelPath, classifyModelPath)
    # runMultipleVideos(rootPath + 'DatasetFeb10/MIA/',date, modelPath, classifyModelPath)

    runMultipleVideos(rootPath + 'VideoInferences/DatasetFeb10/MIA/',name, dateFolder, modelPath, classifyModelPath)
    runMultipleVideos(rootPath + 'VideoInferences/DatasetFeb10/HPNE/',name, dateFolder, modelPath, classifyModelPath)
    runMultipleVideos(rootPath + 'VideoInferences/Cancer Cells February 13/1_9 Ratio/',name, dateFolder, modelPath, classifyModelPath)
    runMultipleVideos(rootPath + 'VideoInferences/Cancer Cells February 13/99_1 Ratio/',name, dateFolder, modelPath, classifyModelPath)

    runMultipleVideos(rootPath + 'VideoInferences/DatasetDec15/HPNE/',name, dateFolder, modelPath, classifyModelPath)
    runMultipleVideos(rootPath + 'VideoInferences/DatasetDec15/MIA/',name, dateFolder, modelPath, classifyModelPath)


    # runMultipleVideos(rootPath + 'VideoInferences/TestOutput/',name, dateFolder, modelPath, classifyModelPath)