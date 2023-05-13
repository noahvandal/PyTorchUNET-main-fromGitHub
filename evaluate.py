
import torch
from dataset import color2label, cellColor2Label
from PIL import Image
import tqdm
import numpy as np
from dataset import getDataset, rgbToOnehotNew
from torchvision import transforms
from model import UNET
from auxiliary import saveImage, pasteTextOnImage, determineROI, sparseToRGB



if torch.cuda.is_available():
    device = 'cuda:0'
    print('Running on the GPU')
else:
    device = 'cpu'
    print('Running on the CPU')

imgHeight = 960
imgWidth = 1280
date = '020423'

rootPath = 'C:/Users/noahv/OneDrive/NDSU Research/Microfluidic ML Device/Videos/Train Images/3D Train/FineTuneFullTrain/Dataset/'

MODEL_PATH = rootPath + '020423_fulltrain_2x_4c_ReduceLR_FineTune_model.pt'
# use test images for testing
IMAGE_DIR = rootPath + 'Test'
SAVE_PATH = rootPath + 'Test/Output_RLRTestFT020523/'

CSV_SAVE = rootPath + 'Test/Output_RLRTestFT020523.csv'
doISaveCSV = True  # write csv data to file or not

classList = ['HPNE', 'MIA', 'PS']
# classList = ['HPNE', 'MIA', 'PS Bead', ]

roiData = []


def save_predictions(data, model, savePath, date):
    # model.eval()
    with torch.no_grad():
        # data = tqdm(data)
        for idx, batch in enumerate(data):

            # here 's' is the name of the file stored in the root directory
            X, y, s = batch
            X, y = X.to(device), y.to(device)

            ## inference predictions
            predictions = model(X)

            imgs = onehotToRGB(predictions, color2label)

            s = str(s)
            pos = s.rfind('/', 0, len(s))
            name = s[pos + 1:-7]

            imgname = name + '_' + date + 'rgb'

            origImg = X.detach().cpu().numpy()
            origImg = origImg[0, :, :, :]
            origImg = np.transpose(origImg, [1, 2, 0])

            y = y.detach().cpu().numpy()

            ## since mask is loaded from dataset as one-hot, must convert back to rgb for saving
            y = onehotENCODEToRGB(y, color2label)

            y = y[:, :, ::-1]  # for some reason dimension needs to be flipped
            origImg = origImg[:, :, ::-1]


            classiou, totalroi = determineROI(imgs, y, path=False)

            # for cancer cells only (no PS):
            roiData.append(
                [name, classiou[0], classiou[1], classiou[2], totalroi])
            classIndex = np.argmax([classiou[0], classiou[1]])

            roi = str(classiou[classIndex])

            roiText = classList[classIndex] + ': ' + roi

            imgs = pasteTextOnImage(imgs, str(roiText))

            saveImage(origImg, name, savePath + '/', imgs)

            if doISaveCSV:
                roiarray = np.array(roiData)
                np.savetxt(CSV_SAVE, roiarray, fmt="%s", delimiter=',',
                           header='ImgName,HPNE IoU,MIA IoU,PS IoU,Background IoU,Total IoU')

            print('saved image to :', savePath + imgname)




def onehotToRGB(OH, colorDict):
    onehot = np.array(OH)
    onehot = np.argmax(onehot, axis=1)  # input: (1,4,h,w) output: (1,h,w)
    output = np.zeros((3,) + onehot.shape[1:3])
    onehot = np.transpose(onehot, [1, 2, 0])
    output = np.transpose(output, [1, 2, 0])

    for label, color in enumerate(colorDict.keys()):
        if label < len(colorDict.keys()):
            ## for each label, set the output image to the corresponding color
            output[np.all(onehot == label, axis=-1)] = color

    return output


def onehotENCODEToRGB(OH, colorDict):
    onehot = np.array(OH)
    output = np.zeros((3,) + onehot.shape[1:3])
    onehot = np.transpose(onehot, [1, 2, 0])
    output = np.transpose(output, [1, 2, 0])

    for label, color in enumerate(colorDict.keys()):
        if label < len(colorDict.keys()):
            output[np.all(onehot == label, axis=-1)] = color

    return output


def evaluate(modelPath, imgPath, savePath, date):
    transform = transforms.Compose([
        transforms.Resize((imgHeight, imgWidth),
                          interpolation=Image.NEAREST)
    ])

    trainSet = getDataset(imgPath, transform, eval=True)

    print('Data has been loaded!')

    net = UNET(in_channels=3, classes=4).to(device)
    checkpoint = torch.load(modelPath, map_location=torch.device(device))
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    print(f'{modelPath} has been loaded and initialized')
    save_predictions(trainSet, net, savePath, date)


if __name__ == '__main__':
    evaluate(MODEL_PATH, IMAGE_DIR, SAVE_PATH, date)
