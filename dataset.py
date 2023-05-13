import torch
from collections import namedtuple
import os
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

# create labels for each of the classes, resepective to the color scheme given for mask creation
Label = namedtuple('Label', ['name', 'id', 'color'])
# labels = [Label('HPNE', 0, (0, 255, 255)),
#           Label('MIA', 1, (255, 0, 255)),
#           Label('PSBead', 2, (255, 255, 0)),
#           Label('Background', 3, (255, 255, 255))]

labels = [Label('HPNE', 0, (0, 255, 255)),
          Label('MIA', 1, (255, 0, 255)),
          Label('PSBead', 2, (255, 255, 0)),
          Label('Background', 3, (255, 255, 255))]


cellLabels = [Label('HPNE', 0, (0, 255, 255)),
              Label('MIA', 1, (255, 0, 255)),
              #   Label('PSBead', 2, (255, 255, 0)),
              Label('Background', 2, (255, 255, 255))]

name2label = {label.name: label for label in labels}
id2label = {label.id: label for label in labels}
color2label = {label.color: label for label in labels}
cellColor2Label = {label.color: label for label in cellLabels}

# return dataset when called upon


class Dataset(Dataset):
    def __init__(self, rootDir, transform=None, eval=False):
        self.transform = transform
        self.maskList = []
        self.imgList = []
        self.eval = eval

        # self.maskPath = os.path.join(os.getcwd(), rootDir + '/Images/')
        # self.imgPath = os.path.join(os.getcwd(), rootDir + '/Masks/')
        self.maskPath = os.path.join(rootDir + '/Masks/')
        self.imgPath = os.path.join(rootDir + '/Images/')

        # print('maskpath', self.maskPath, self.imgPath)

        imgItems = os.listdir(self.imgPath)
        # print(imgItems)

        maskItems = [rootDir + '/Masks/' + path for path in imgItems]
        imgItems = [rootDir + '/Images/' + path for path in imgItems]

        self.maskList.extend(maskItems)
        self.imgList.extend(imgItems)
        # print(self.maskList)
        # print(self.imgList)

    def __len__(self):
        length = len(self.imgList)
        return length

    def __getitem__(self, index):
        imgPath = self.imgList[index]
        maskPath = self.maskList[index]

        img = Image.open(imgPath)
        mask = Image.open(maskPath)

        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform(mask)

        img = transforms.ToTensor()(img)
        # mask = transforms.ToTensor()(mask)
        # mask = np.array(mask)
        # print(mask.shape)
        # mask = np.transpose(mask, [1, 2, 0])
        mask = rgbToOnehotNew(mask, cellColor2Label)
        # print('mask shape', mask.shape)

        # mask = mask[2, :, :]  # removing the alpha channel (not really needed)
        mask = torch.from_numpy(mask)
        # mask = torch.from_numpy(mask)
        mask = mask.type(torch.LongTensor)

        if self.eval:
            return img, mask, self.imgList[index]
        else:
            return img, mask

        return img, mask

# getting datset using dataload function and native functions from torch


def getDataset(rootDir, transform=None, batchSize=1, shuffle=True, pin_memory=True, eval=False):
    data = Dataset(rootDir=rootDir, transform=transform, eval=eval)
    dataLoaded = torch.utils.data.DataLoader(data, batch_size=batchSize,
                                             shuffle=shuffle, pin_memory=pin_memory)

    return dataLoaded

# saving images when called upon


def saveImages(tensor_pred, folder, image_name):
    tensor_pred = transforms.ToPILImage()(tensor_pred.byte())
    filename = f"{folder}\{image_name}.png"
    tensor_pred.save(filename)

# function that is called when training model


def trainFunction(data, model, optimizer, lossFunction, device):
    data = tqdm(data)
    for index, batch in enumerate(data):
        X, y = batch
        X, y = X.to(device), y.to(device)

        predictions = model(X)

        loss = lossFunction(predictions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()



def rgbToOnehotNew(rgb, colorDict):
    rgb = np.array(rgb).astype('int32')
    dense = np.zeros(rgb.shape[:2])
    for label, color in enumerate(colorDict.keys()):
        color = np.array(color)
        pixel = np.zeros(len(colorDict.keys()))
        pixel[label] = 1 ## converting to one-hot encoding instead of sparse.

        ## for each class present, match the color and set the pixel to the class label
        if label < len(colorDict.keys()):
            dense[np.all(rgb == color, axis=-1)] = pixel

    return dense

## instead of usign one-hot encoding, use sparse encoding
def rgbToOnehotSparse(rgb, colorDict):
    rgb = np.array(rgb).astype('int32')
    dense = np.zeros(rgb.shape[:2])
    for label, color in enumerate(colorDict.keys()):
        color = np.array(color)
        if label < len(colorDict.keys()):
            dense[np.all(rgb == color, axis=-1)] = label

    return dense


def outputClassImages(onehot, color_dict):
    onehot = np.array(onehot)
    onehot = np.argmax(onehot, axis=1)
    onehot = np.transpose(onehot, [1, 2, 0])

    # numClasses = len(color_dict.keys())
    outputList = []  # temporarily store each image as it is created
    totalOutput = 255*np.ones(onehot.shape[0:2]+(3,))

    for i, k in enumerate(color_dict.keys()):
        output = 255*np.zeros(onehot.shape[0:2])
        if i < len(color_dict.keys()) - 1:  # excluding background class (last entry in list)
            totalOutput[np.all(onehot == i, axis=2)] = k
            output[np.all(onehot == i, axis=-1)] = 255
            outputList.append(output)

    outputList.append(totalOutput)
    return outputList


## instead of modifying class based output, will simply convert to bw whenever a non-background class is present. Used this when transferring from MC to binary
def RGBtoBW(image, isGrayscale=False):
    image = image.astype('uint8')
    bw = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype('uint8')
    ret, bw = cv2.threshold(bw,254,255,cv2.THRESH_BINARY)  ## threshold to combine all colors in output channel
    
    if isGrayscale:
        bw = np.expand_dims(bw, axis=-1)  ## making 3-d tupule
        bw = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)
    
    return bw