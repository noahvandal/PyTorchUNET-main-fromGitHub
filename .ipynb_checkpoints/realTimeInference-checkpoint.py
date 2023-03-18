from preProcess import preProcess
from contourEllipseDetection import ellipseDetection, processOneHotIntoClassBW, getEllipsesFromClassList
from tracking import Tracking, postVideoProcessList
import torch
import cv2
import csv
import numpy as np
from model import UNET
from IPython.display import clear_output
from auxiliary import saveImageOutput, putEllipsesOnImage, suppressUndesirableEllipses
from dataset import cellColor2Label, color2label, outputClassImages
from evaluate import onehot_to_rgb
from torchvision import transforms
from tracking import Tracking
import time

if torch.cuda.is_available():
    DEVICE = 'cuda:0'
else:
    DEVICE = 'cpu'


IMAGE_SHAPE = (960, 1280)


def main(videoPath, saveImagePath, saveCSVPath, modelPath, magnification):

    headerFile = ['ID', 'Coordinates', 'Axes Length', 'Number of Frames Present',
                  'Average Speed (px)', 'Identity', 'Number of frames as identity', 'Diameter']  # data for csv output

    model = UNET(in_channels=3, classes=4)
    checkpoint = torch.load(modelPath, map_location=torch.device(DEVICE))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f'{modelPath} has been loaded and initialized')

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

        # origImg = np.resize(frame,[960, 1280], interpolation=cv2.INTER_CUBIC)
        print(frame.shape)
        origImg = cv2.resize(frame, [IMAGE_SHAPE[1], IMAGE_SHAPE[0]],
                             interpolation=cv2.INTER_LINEAR)  # have to flip dimensions due to being cv2 ** uggghhh***

        totalEllipses = []

        # frame = preProcess(frame)  # pre-process image
        # frame = frame.preProcessImage()
        print(origImg.shape)
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
        allimgs = outputClassImages(output, color2label)
        # print(output)
        totalOutputImage = onehot_to_rgb(output, color2label)
        # imgs = allimgs[0]
        # allimgs.append(totalOutput)

        # print(output)

        totalEllipses = getEllipsesFromClassList(allimgs, nameList)

        # print(iterator)

        tracker = Tracking(CurrentDict, GlobalList, totalEllipses,
                           id, pixelChange, magnification)

        CurrentDict, id, GlobalList, _, changeinpixel = tracker.comparePointsList()

        # GlobalList.extend(outOfFrameList)
        # print('global list', GlobalList)
        # print('pixel change', changeinpixel)

        # calculating change in pixel value from frame to frame; can determine flow rate from this.
        pixelChange = (pixelChange*iterator + changeinpixel) / iterator

        print(CurrentDict)
        imgs = putEllipsesOnImage(totalOutputImage, CurrentDict, magnification)
        print(origImg.shape, imgs.shape)
        saveImageOutput(origImg, imgs, str(iterator) + '_All',
                        saveImagePath, doISave=True, showOutput=False)  # ensure save is set to true to save output

        print('Global List Before', GlobalList)
        listcompare = postVideoProcessList(GlobalList)
        GlobalList = listcompare.SingleComparePointsGlobal()
        print('Global List', GlobalList)

        endFrame = time.time()

        clear_output(wait=True)
        print('Frame Number: ', iterator)
        print('Frame Time: ', endFrame-startFrame)
        print('Inference Time: ', inferenceEnd - inferenceStart)

        iterator += 1

        # clear_output(wait=True)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

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
    cv2.destroyAllWindows()

    with open(saveCSVPath, 'w', newline='') as csvwriter:
        write = csv.writer(csvwriter)
        write.writerows(GlobalList)


if __name__ == '__main__':

    rootPath = 'C:/Users/noahv/OneDrive/NDSU Research/Microfluidic ML Device/'

    saveImagePath = rootPath + 'Videos/August 2022/VideosAllTogether/20um/5_25_640_0/'
    saveCSVPath = rootPath + 'Videos/August 2022/VideosAllTogether/20um/0.csv'

    # define model path here
    # modelPath = rootPath + \
    # 'Videos/Train Images/3D Train/FullTrainDataset/Weights/01182023_ReduceLRonPlateau_model_2xscale.pt'
    modelPath = 'C:/Users/noahv/OneDrive/NDSU Research/Microfluidic ML Device/Videos/Train Images/3D Train/FineTuneFullTrain/Dataset/020423_fulltrain_2x_4c_ReduceLR_FineTune_model.pt'
    videoPath = rootPath + 'Videos/August 2022/VideosAllTogether/20um/5_25_640_0.mp4'
    # videoPath = rootPath + \
    # 'Videos/August 2022/20um/5ul_min/25x mag/1280x960 px/5_25_1280_0.mp4'
    magnification = 25

    main(videoPath, saveImagePath, saveCSVPath,
         modelPath, magnification=magnification)
