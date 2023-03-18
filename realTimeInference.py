from preProcess import preProcess
from contourEllipseDetection import ellipseDetection, processOneHotIntoClassBW, getEllipsesFromClassList
from tracking import Tracking, postVideoProcessList
import torch
import cv2
import csv
from sys import stdout
import numpy as np
from model import UNET
from IPython.display import clear_output
from auxiliary import saveImageOutput, putEllipsesOnImage, suppressUndesirableEllipses
from dataset import cellColor2Label, color2label, outputClassImages
from evaluate import onehot_to_rgb
from torchvision import transforms
from newTracking import Tracking
import time
import os

if torch.cuda.is_available():
    DEVICE = 'cuda:1'
else:
    DEVICE = 'cpu'


IMAGE_SHAPE = (960, 1280)


def main(videoPath, saveImagePath, saveCSVPath, modelPath, magnification):

    headerFile = ['ID', 'Coordinates', 'Axes Length', 'Number of Frames Present',
                  'Average Speed (px)', 'Identity', 'Number of frames as identity', 'Diameter']  # data for csv output

    model = UNET(in_channels=3, classes=3)
    checkpoint = torch.load(modelPath, map_location=torch.device(DEVICE))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f'{modelPath} has been loaded and initialized')

    model = model.to(DEVICE)

    # model = torch.load(modelPath).to(DEVICE)

    GlobalList = []
    CurrentDict = {}  # list of all saved points that were inferenced in frames

    id = 0

    nameList = ['HPNE', 'MIA', 'PSBead', 'All']
    # magnification = 25

    pixelChange = 1  # just tracking movement of pixels through frames
    magnification = 25  # should be set for entire frame

    # loop for each frame
    iterator = 1  # count how many frames pass

    src = cv2.VideoCapture(videoPath)
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

        allimgs = outputClassImages(output, cellColor2Label)  ## getting ellipses from each class list 

        # getting actual color mapping of all classes present
        totalOutputImage = onehot_to_rgb(output, cellColor2Label)

        # getting ellipses for each class present
        totalEllipses = getEllipsesFromClassList(allimgs, nameList)

        tracker = Tracking(CurrentDict, GlobalList, totalEllipses,
                           id, pixelChange, magnification)   ## update list based on points in previous frame. 

        CurrentDict, id, GlobalList, _, changeinpixel = tracker.comparePointsList()

        # GlobalList.extend(outOfFrameList)
        # print('global list', GlobalList)
        # print('pixel change', changeinpixel)

        # calculating change in pixel value from frame to frame; can determine flow rate from this.
        pixelChange = (pixelChange*iterator + changeinpixel) / iterator

        # print(CurrentDict)
        imgs = putEllipsesOnImage(totalOutputImage, CurrentDict, magnification)
        print(origImg.shape, imgs.shape)
        saveImageOutput(origImg, imgs, str(iterator) + '_All',
                        saveImagePath, doISave=True, showOutput=False)  # ensure save is set to true to save output

        # print('Global List Before', GlobalList)
        listcompare = postVideoProcessList(GlobalList)
        GlobalList = listcompare.SingleComparePointsGlobal()
        # print('Global List', GlobalList)

        endFrame = time.time()

        clear_output(wait=True)
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



def runMultipleVideos(rootPath, extraName, modelPath):
    videoList = os.listdir(rootPath)
    for video in videoList:
        videoName = video[:-4]

        saveImagePath = rootPath + str(videoName) + extraName + '/'
        doesPathExist = os.path.exists(saveImagePath)

        if doesPathExist == False:
            os.mkdir(saveImagePath)
        
        saveCSVPath = rootPath + str(videoName) + extraName + '.csv'

        videoPath = rootPath + video

        print(videoPath)

        magnification = 25

        main(videoPath, saveImagePath, saveCSVPath, modelPath, magnification)


if __name__ == '__main__':

    rootPath = '/home/noahvandal/my_project_dir/my_project_env/UNET_Color/'

    # saveImagePath = rootPath + 'DatasetFeb10/OutputImages/HPNE_230203121917/'
    # saveCSVPath = rootPath + 'DatasetFeb10/HPNE_230203121917.csv'

    # define model path here
    # modelPath = rootPath + \
    # 'Videos/Train Images/3D Train/FullTrainDataset/Weights/01182023_ReduceLRonPlateau_model_2xscale.pt'

    modelPath = rootPath + 'UNET_MC_PyTorch/FineTuneModels/021123_2x_3c_Train_model.pt'
    # videoPath = rootPath + 'DatasetFeb10/HPNE/230203121917.mp4'
    # videoPath = rootPath + \
    # 'Videos/August 2022/20um/5ul_min/25x mag/1280x960 px/5_25_1280_0.mp4'
    # magnification = 25
    date = '_3c_teset_021222_'

    # main(videoPath, saveImagePath, saveCSVPath,
        #  modelPath, magnification=magnification)
    runMultipleVideos(rootPath + 'DatasetDec15Cells/HPNE/',date, modelPath)
    runMultipleVideos(rootPath + 'DatasetDec15Cells/MIA/',date, modelPath)
