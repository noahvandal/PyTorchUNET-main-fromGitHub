## this script runs the classification portion of the model. 
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision
import os
import pandas as pd
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from IPython.display import clear_output

if torch.cuda.is_available():
    device = 'cuda:1'
else:
    device = 'cpu'


torch.manual_seed(43)
np.random.seed(43)
random.seed(43)


class Classifier_v2(nn.Module):
    def __init__(self, batchsize):
        super(Classifier_v2, self).__init__()
        self.batchsize = batchsize

        self.conv1 = torch.nn.Conv2d(3, 6, 3)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 12, 3)
        self.conv3 = torch.nn.Conv2d(12, 24, 3)
        self.fc1 = torch.nn.Linear(24 * 2 * 2, 100)
        self.fc2 = torch.nn.Linear(100, 10)
        self.fc3 = torch.nn.Linear(10, 2)

        self.drop1 = torch.nn.Dropout(0.1)
        self.drop2 = torch.nn.Dropout(0.2)
        self.drop3 = torch.nn.Dropout(0.3)

        self.dropFC = torch.nn.Dropout(0.5)

        self.batchnorm1 = torch.nn.BatchNorm2d(6)
        self.batchnorm2 = torch.nn.BatchNorm2d(12)
        self.batchnorm3 = torch.nn.BatchNorm2d(24)

    
    def computeUnit(self, x, conv, batchnorm, pool, dropout):
        x = conv(x)
        x = batchnorm(x)
        x = self.relu(x)
        x = pool(x)
        x = dropout(x)
        return x

    def forward(self, x):
        x = self.computeUnit(x, self.conv1, self.batchnorm1, self.pool, self.drop1)
        x = self.computeUnit(x, self.conv2, self.batchnorm2, self.pool, self.drop2)
        x = self.computeUnit(x, self.conv3, self.batchnorm3, self.pool, self.drop3)

        # self.fc1 = torch.nn.Linear(24 * x.shape[2] * x.shape[3], 100)

        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.dropFC(x)
        x = self.relu(self.fc2(x))
        x = self.dropFC(x)
        x = self.fc3(x)

        return x


class Classify(nn.Module):
    def __init__(self, batchsize):
        super(Classify, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  ## input channel, output channel, filter size
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2) ## poolsize, stride
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 2)
        self.batchsize = batchsize
    
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x = torch.flatten(x)
        x = x.view(-1,16*5*5)
        # x = x.view(16*5*5,-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def createDataset(path):  #input path to folder containing train; ensure all samples have name corresponding to class in them. 
    print('path',path)
    datalist = os.listdir(path)
    allpaths = []
    for data in datalist:
        classtype = isClass(data)
        allpaths.append([path + data, classtype])
    allpaths = pd.DataFrame(allpaths, columns=["Image", "Class"])
    return allpaths

def getDataset(path, batchsize,numsteps,isVal):
    imgdata = dataGenerator(path, batchsize,numsteps,isVal)
    loadedData = DataLoader(imgdata,batchsize,shuffle=False, drop_last=True)
    return loadedData

def isClass(str): ## given string, is the class contained in the name? 
    stringClass = ""
    if "HPNE"  in str:
        stringClass = [1,0]
    if 'MIA' in str:
        stringClass = [0,1]
    return stringClass


class dataGenerator(Dataset):
    def __init__(self, dataframe, batchsize, numsteps, isVal = False):
        self.dataframe = dataframe
        self.batchsize = batchsize
        self.numsteps = numsteps
        self.Validation = isVal
        self.resize = (32,32)

    
    def __len__(self):
        return len(self.dataframe) // self.batchsize

    def on_epoch_end(self):
        self.dataframe = self.dataframe.reset_index(drop = True)

    def __getitem__(self, index):
        images = []
        labels = []

        img = cv2.imread(self.dataframe["Image"][index])
        img = cv2.resize(img,self.resize)
        label = np.array(self.dataframe["Class"][index])

        images.append(img)
        labels.append(label)

        img = np.transpose(img, [2, 0, 1])
        img = torch.from_numpy(img)
        img = img.float()

        label = torch.from_numpy(label)
        label = label.float()


        if self.Validation:
            return (img, label, self.dataframe['Image'][index])

        else:
            return (img, label)
        

def checkAccuracy(tsr1, tsr2, classesPresent, printOutput):
    areEqual = False
    accCount = 0
    accuracy = 0
    compareList = []

    if len(tsr1) == len(tsr2):
        if printOutput:
            print('equal!!')
        areEqual = True

    if areEqual:
        for i in range(0,len(tsr1)):
            if tsr1[i] == tsr2[i]:
                if printOutput:
                    print('Equal!', classesPresent[tsr1[1]])
                accCount += 1
            if tsr1[i] != tsr2[i]:
                if printOutput:
                    print('Unequal: Predict:', classesPresent[tsr1[i]], 'Actual:', classesPresent[tsr2[i]])
            
            compareList.append([classesPresent[tsr1[i]],classesPresent[tsr2[i]]])
        
        accuracy = accCount / len(tsr1)

    
    return accuracy, compareList


def datasetAcquirerShuffler(srcPath, numTrain, numVal):  ## input path to dataset, and will split into test and train by itself, using random shuffle.
    allFiles = os.listdir(srcPath)

    random.shuffle(allFiles) ## where the shuffle takes place!!

    trainPaths = []
    valPaths = []
    for i, data in enumerate(allFiles):
        if i < numTrain:
            classtype = isClass(data)
            trainPaths.append([srcPath + data, classtype])
        if numTrain <= i < (numTrain + numVal):
            classtype = isClass(data)
            valPaths.append([srcPath + data, classtype])
    trainPaths = pd.DataFrame(trainPaths, columns=["Image", "Class"])
    valPaths = pd.DataFrame(valPaths, columns = ['Image','Class'])
    return trainPaths, valPaths




def testFunction(testPath,classes, modelPath, csvSave):
    testlist = os.listdir(testPath)

    outputList = []
    runningAvgAccCount = 0
    runningAvgAcc = 0

    model = Classify(1).to(device) ## batch size of 1
    checkpoint = torch.load(modelPath)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Model Successfully Loaded')

    for i, test in enumerate(testlist):
        label = np.array(isClass(test))
        img = cv2.imread(testPath + test)
        img = cv2.resize(img, (32,32))
        img = np.transpose(img, [2, 0, 1])
        img = np.expand_dims(img, 0)

        label = np.expand_dims(label,0)


        img = torch.from_numpy(img)
        label = torch.from_numpy(label)

        img = img.float()
        label = label.float()
        
        with torch.no_grad():
            img = img.to(device)
            label = label.to(device)
            outputs = model(img)
        
        _, index = torch.max(outputs, dim=1)
        _,labelIndex = torch.max(label, dim=1) 

        ## see whether each image was correct or not. 
        acc, comparelist = checkAccuracy(index,labelIndex,classes, printOutput = False)

        ## get running average of the accuracy
        runningAvgAccCount += acc
        runningAvgAcc = runningAvgAccCount / (i + 1)

        print(len(comparelist), comparelist)
        outputList.append([test,acc,comparelist[0][0], comparelist[0][1], runningAvgAcc])

        if i%20 == 0:
            print(i)
        
    print(type(outputList), len(outputList))
    outputList = np.array(outputList)
    np.savetxt(csvSave, outputList, delimiter=',',header='',fmt='%s')
    print('All images tested')



def rollingAverage(inputlist, window):
    if len(inputlist) >= window:
        while len(inputlist) > window:
            inputlist.pop(0)
    
    avg = sum(inputlist) / len(inputlist)
    return avg

def plotTrain(data, isList):
    if isList:
        data = np.array(data)

    fig, ax = plt.subplots()

    
    ax.plot(data[:,0], label = 'Train Loss')
    ax.plot(data[:,1], label = 'Val Loss')

    ax2 = ax.twinx()

    ax2.plot(data[:,2], label = 'Train Accuracy')
    ax2.plot(data[:,3], label = 'Val Accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Loss/Accuracy')
    plt.title('Training Loss/Accuracy')
    plt.legend()
    plt.show()




def trainFunction(trainPath, valPath, sourcePath, modelPath, csvSave):
    batchsize = 16
    model = Classifier_v2(batchsize)
    model.to(device)
    loadModel = False

    classes = ['HPNE', 'MIA']

    lr = 0.00001
    EPOCHS = 40000
    epoch = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr, betas=(0.9,0.999))

    batchsize = 16
    numsteps = 4
    valbatchsize = 4
    valnumsteps = 2

    ## initialize datasets first just for delcaration of variables and in case shuffle doesnt work. Otherwise will use shuffle dataset. 
    
    trainDataSet = createDataset(trainPath)
    trainSet = getDataset(trainDataSet,batchsize,numsteps,isVal=False)

    valDataSet = createDataset(valPath)
    valSet = getDataset(valDataSet,valbatchsize,valnumsteps,isVal=True)

    if loadModel:
        checkpoint = torch.load(modelPath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        epoch = checkpoint['epoch'] + 1
        # epoch = 0 ### making 0 so it will run again' 
        lossVals = checkpoint['loss_values']
        print('Model Successfully Loaded')

    trainValues = []

    rollingAvgAccTrain = []
    rollingAvgLossTrain = []
    rollingAvgAccVal = []
    rollingAvgLossVal = []
     
    
    for e in range(epoch, EPOCHS):
        tLoss = 0
        vLoss = 0
        tAcc = 0
        vAcc = 0
        acc = 0

        ## manual weight decay
        if e % 1000 == 0:  
            lr = lr * 0.95 ## reduce lr
            optimizer = optim.Adam(model.parameters(), lr, betas=(0.9,0.999))


        for i, data in enumerate(trainSet):
            inputs, label = data
            optimizer.zero_grad()

            inputs = inputs.to(device)
            label = label.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step() ## decay if necessary for adam
            tLoss += loss
            outLoss, index = torch.max(outputs, dim=1)
            _,labelIndex = torch.max(label, dim=1)

            ## finding train accuracy
            acc, _ = checkAccuracy(index,labelIndex,classes, printOutput = False)
            tAcc += acc
        
        tAcc = tAcc / (i + 1)


        for i, data in enumerate(valSet):
            inputs, label, index = data

            inputs = inputs.to(device)
            label = label.to(device)
            outputs = model(inputs)

            vLoss += criterion(outputs, label)
            outLoss, index = torch.max(outputs, dim=1)
            _,labelIndex = torch.max(label, dim=1)

            ## finding val accuracy
            acc, _ = checkAccuracy(index,labelIndex,classes, printOutput = False)
            vAcc += acc
        
        vAcc = vAcc / (i + 1)

        tLoss = tLoss.cpu().detach().numpy()
        vLoss = vLoss.cpu().    detach().numpy()


        rollingAvgAccTrain.append(tAcc)
        rollingAvgLossTrain.append(tLoss)
        rollingAvgAccVal.append(vAcc)
        rollingAvgLossVal.append(vLoss)

        ## getting rolling average accuracy of last n epochs
        window = 200

        rTacc = rollingAverage(rollingAvgAccTrain, window)
        rTloss = rollingAverage(rollingAvgLossTrain, window)
        rVacc = rollingAverage(rollingAvgAccVal, window)
        rVloss = rollingAverage(rollingAvgLossVal, window)


        trainValues.append([tLoss,vLoss,tAcc, vAcc])

        clear_output(wait=True)

        print(len(rollingAvgAccTrain))

        plotTrain(trainValues, True)

        print('One epoch down! here is the loss:', tLoss, vLoss)
        print('Here is the rolling average loss: {:.2f}, {:.2f}'.format(rTloss, rVloss))
        print('Here is the accuracy: {:.2f}, {:.2f}'.format(tAcc, vAcc))
        print('Here is the ra accuracy: {:.2f}, {:.2f}'.format(rTacc, rVacc))
        print('Epoch number: ', e)
        # for loss in lossVals:
            # print(loss)
        
        torch.save({
            'model_state_dict': model.state_dict(),  ## these are the weights and overall configuration
            'optim_state_dict': optimizer.state_dict(),
            'epoch': e,
            'loss_values': trainValues
        }, modelPath)

        if e % 1000 == 0:
            np.savetxt(csvSave, trainValues, delimiter=',',header='Train Loss,Val Loss,Train Accuracy,Val Accuracy')



if __name__ == '__main__':
    rootPath = '/home/noahvandal/my_project_dir/my_project_env/UNET_Color/HybridNet/'

    trainPath = rootPath + 'Dataset/Train/'
    valPath = rootPath + 'Dataset/Val/'
    testPath = rootPath + 'Dataset/Source/Test/'

    sourcePath = rootPath + 'Dataset/Source/AllTrain/'

    csvSave = rootPath + 'LearnData/'

    modelSrc = rootPath + 'Dataset/Model/'
    modelName = '032323_2c_v2_shuffle10e_reducelr1000e_095_pureTest'
    modelPath = modelSrc + modelName +'.pt'

    csvSave = rootPath + 'Dataset/LearningData/' + modelName + '.csv'
    csvSaveTest = rootPath + 'Dataset/LearningData/' + modelName  + '_TestData.csv'

    classes = ['HPNE','MIA']


    if not os.path.exists(modelSrc):
        os.makedirs(modelSrc)
    
    trainFunction(trainPath,valPath,sourcePath, modelPath, csvSave)

    testFunction(testPath,classes,modelPath,csvSaveTest)