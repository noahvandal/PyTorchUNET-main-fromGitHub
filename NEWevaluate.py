import torch
import numpy as np
import time
import os
import pandas as pd
from model import UNET
import cv2
from torch.utils.data import DataLoader, Dataset
from dataset import color2label
from auxiliary import showImage, saveImage

if torch.cuda.is_available():
    DEVICE = 'cuda:0'
else:
    DEVICE = 'cpu'

ImageSize = (960, 1280)

rootPath = 'C:/Users/noahv/OneDrive/NDSU Research/Microfluidic ML Device/Videos/Train Images/3D Train/FineTuneFullTrain/'

modelPath = rootPath + 'Dataset/020423_fulltrain_2x_4c_ReduceLR_FineTune_model.pt'
# use test images for testing
imgPath = rootPath + 'Dataset/Test/'
savePath = rootPath + 'Dataset/Test/Output_RLRTestFT020523/'


doIwanttoshowImages = False
saveImagestoFile = True


def createDataset(rootPath):
    imglist = os.listdir(rootPath + 'Images/')
    allPaths = []
    for img in imglist:
        imgPath = rootPath + 'Images/' + img
        maskPath = rootPath + 'Masks/' + img
        allPaths.append([imgPath, maskPath])
    outputDataset = pd.DataFrame(allPaths, columns=['Images', 'Masks'])
    return outputDataset


def onehotToRGB(OH, colorDict):
    onehot = OH.clone().detach().cpu()  # necessary to remove from gpu to
    onehot = np.array(onehot)
    # print(onehot.shape)
    onehot = np.argmax(onehot, axis=1)  # input: (1,4,h,w) output: (1,h,w)
    output = np.zeros((3,) + onehot.shape[1:3])
    onehot = np.transpose(onehot, [1, 2, 0])
    output = np.transpose(output, [1, 2, 0])
    # print(onehot.shape)
    print(onehot.shape, output.shape)
    for label, color in enumerate(colorDict.keys()):
        print(label, color)
        if label < len(colorDict.keys()):
            output[np.all(onehot == label, axis=-1)] = color

    return output

# Image data generator class


def RGBtoOnehot(mask, classDict):

    numClasses = len(classDict.keys())
    onehot = np.zeros((mask.shape[:2]) + (numClasses,)).astype('uint8')
    for label, color in enumerate(classDict.keys()):
        if label < len(classDict.keys()):
            pixel = np.zeros(numClasses)
            pixel[label] = 1
            onehot[np.all(mask == color, axis=-1)] = pixel
    return onehot


class ImageDataGenerator(Dataset):

    def __init__(self, dataframe, batch_size, num_steps, n_classes=4, augment=False, isVal=False):
        self.dataframe = dataframe
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.augment = augment
        self.stepSize = num_steps
        self.Validation = isVal

    def __len__(self):
        return len(self.dataframe) // self.batch_size

    def on_epoch_end(self):
        self.dataframe = self.dataframe.reset_index(drop=True)

    def __getitem__(self, index):

        processed_images = []
        processed_masks = []

        # for _ in range(self.stepSize):

        the_image = cv2.imread(self.dataframe['Images'][index])
        the_mask = cv2.imread(self.dataframe['Masks'][index]).astype('uint8')
        one_hot_mask = RGBtoOnehot(the_mask, color2label)

        processed_image = cv2.resize(
            the_image, (ImageSize[1], ImageSize[0])) / 255.0
        # for some reason, cv2.resize takes the input as flipped axes
        processed_mask = cv2.resize(one_hot_mask, (ImageSize[1], ImageSize[0]))

        processed_images.append(processed_image)
        processed_masks.append(processed_mask)

        # batch_x = np.array( processed_images)
        # batch_y = np.array( processed_masks)

        processed_image = np.transpose(processed_image, [2, 0, 1])
        processed_mask = np.transpose(processed_mask, [2, 0, 1])

        processed_image = torch.from_numpy(processed_image)
        processed_mask = torch.from_numpy(processed_mask)

        processed_image = processed_image.float()
        processed_mask = processed_mask.float()

        # processed_image = processed_image.type(torch.Tensor)
        # processed_mask = processed_mask.type(torch.LongTensor)

        # processed_images.append(processed_image)
        # processed_masks.append(processed_mask)

        # step_x = np.array(processed_images)
        # step_y = np.array(processed_masks)
        if self.Validation:
            return (processed_image, processed_mask, self.dataframe['Images'][index])

        else:
            return (processed_image, processed_mask)


def getDataset(data, batchSize, stepSize, numClasses, shuffle, istestData=False):
    imgdata = ImageDataGenerator(
        data, batch_size=batchSize, num_steps=stepSize, n_classes=numClasses, isVal=istestData)

    loadedData = DataLoader(imgdata, batch_size=batchSize, shuffle=shuffle)

    return loadedData

# imgList = os.listdir(imgPath)

# img = random.choice(imgList)

# img = cv2.imread(imgPath + img)

# print(img.shape)


net = UNET().to(DEVICE)
checkpoint = torch.load(modelPath, map_location=torch.device(DEVICE))
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()

print('Model loaded')

valDataset = createDataset(imgPath)
valImage = getDataset(valDataset, batchSize=1, stepSize=1,
                      numClasses=4, shuffle=False, istestData=True)

waitVal = 4000  # how long to wait between images being displayed (in ms)

with torch.no_grad():  # no training

    for i, data in enumerate(valImage):
        X, y, path = data
        X, y = X.to(DEVICE), y.to(DEVICE)

        img = net(X)

        img = onehotToRGB(img, color2label)

        origImg = X.clone().detach().cpu()
        origImg = np.array(origImg)
        origImg = origImg[0, :, :, :]
        origImg = np.transpose(origImg, [1, 2, 0])

        # print(type(origImg), origImg.shape)

        path = str(path)
        path = path[-16:-7]
        # print(path)

        if doIwanttoshowImages:
            showImage(origImg, waitVal, path)
            showImage(img, waitVal, path)

        if saveImagestoFile:
            time.sleep(1)
            saveImage(origImg, path, savePath, img)
