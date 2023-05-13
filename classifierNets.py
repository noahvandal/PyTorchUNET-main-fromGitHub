import torch 
import torch.nn as nn
import random
import numpy as np
import torch.nn.functional as F

torch.manual_seed(43)
np.random.seed(43)
random.seed(43)



class ClassifierHyperparam_v2(nn.Module):
    def __init__(self):
        super(ClassifierHyperparam_v2, self).__init__()
        self.conv1 = torch.nn.Conv2d(3,18,9)
        self.conv2 = torch.nn.Conv2d(18, 19, 5)

        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)
        self.pool = torch.nn.MaxPool2d(2,2)
        self.dropout = torch.nn.Dropout(0.20)

        self.fc1 = torch.nn.Linear(19 * 12 * 12, 74)
        self.fc2 = torch.nn.Linear(74, 73)
        self.fc3 = torch.nn.Linear(73, 2)

    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        # x = x.view(x.size(0), -1)
        x = torch.nn.Flatten()(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.softmax(self.fc3(x))

        return x
    
class Classifier_v19(nn.Module):
    def __init__(self):
        super(Classifier_v19, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 13)
        self.conv2 = torch.nn.Conv2d(64, 128, 13)

        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.fc1 = torch.nn.Linear(128 * 7 * 7, 50)
        self.fc2 = torch.nn.Linear(50, 2)


        self.convDrop = torch.nn.Dropout(0.1)
        self.dropout = torch.nn.Dropout(0.5)
        self.softmax = torch.nn.Softmax(dim=1)



    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))

        return x


class Classifier_v18(nn.Module):
    def __init__(self):
        super(Classifier_v18, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 13)
        self.conv2 = torch.nn.Conv2d(32, 32, 9)
        self.conv3 = torch.nn.Conv2d(32, 32, 5)

        self.conv4 = torch.nn.Conv2d(32, 64, 3)

        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.fc1 = torch.nn.Linear(32 * 2 * 2, 20)
        self.fc2 = torch.nn.Linear(20, 10)
        self.fc3 = torch.nn.Linear(10, 2)

        self.convDrop = torch.nn.Dropout(0.1)
        self.dropout = torch.nn.Dropout(0.5)
        self.softmax = torch.nn.Softmax(dim=1)

        self.batchnorm1 = torch.nn.BatchNorm2d(128)
        self.batchnorm2 = torch.nn.BatchNorm2d(64)
        self.batchnorm3 = torch.nn.BatchNorm2d(32)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        # print(x.shape)
        x = self.pool(self.relu(self.conv2(x)))
        # print(x.shape)
        x = self.pool(self.relu(self.conv3(x)))
        # print(x.shape)
        # print(x1.shape, x2.shape, x3.shape)
        # x = self.pool(self.relu(self.conv4(x)))

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        # x = self.dropout(x)
        # x = self.softmax(self.fc2(x))



        return x

class Classifier_v16(nn.Module):
    def __init__(self):
        super(Classifier_v16, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 128, 7)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.fc1 = torch.nn.Linear(128 * 14 * 14, 2)
        self.fc2 = torch.nn.Linear(100, 2)

        self.convDrop = torch.nn.Dropout(0.1)
        self.dropout = torch.nn.Dropout(0.5)

        self.batchnorm1 = torch.nn.BatchNorm2d(128)
        self.batchnorm2 = torch.nn.BatchNorm2d(256)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.batchnorm1(x)
        x = self.pool(self.pool(x))

        x = x.view(x.size(0), -1)
        x = self.fc1(x)


        return x

class Classifier_v15(nn.Module):
    def __init__(self):
        super(Classifier_v15, self).__init__()
        self.conv1_0 = torch.nn.Conv2d(3, 32, 7)
        self.conv1_1 = torch.nn.Conv2d(32, 64, 3)
        self.conv2_0 = torch.nn.Conv2d(3, 32, 5)
        self.conv2_1 = torch.nn.Conv2d(32, 64, 3)
        self.conv3_0 = torch.nn.Conv2d(3, 32, 3)
        self.conv3_1 = torch.nn.Conv2d(32, 64, 3)

        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.softmax = torch.nn.Softmax(dim=1)
        self.fc1 = torch.nn.Linear(64 * 14 * 14 + 64 * 14 * 14 + 64 * 13 * 13, 200)
        self.fc2 = torch.nn.Linear(200, 100)
        self.fc3 = torch.nn.Linear(100, 20)
        self.fc4 = torch.nn.Linear(20, 2)
        self.sigmoid = torch.nn.Sigmoid()
        # self.fc2 = torch.nn.Linear(100, 2)

        self.dropout = torch.nn.Dropout(0.5)
        

        self.batchnorm1 = torch.nn.BatchNorm2d(32)
        self.batchnorm2 = torch.nn.BatchNorm2d(32)
        self.batchnorm3 = torch.nn.BatchNorm2d(32)
        self.batchnorm4 = torch.nn.BatchNorm2d(64)
        self.batchnorm5 = torch.nn.BatchNorm2d(64)
        self.batchnorm6 = torch.nn.BatchNorm2d(64)

    def convAndPool(self, x, conv, batchnorm, pool, activation):
        x = activation(conv(x))
        x = batchnorm(x)
        x = pool(x)
        return x


    def forward(self, x):
        x0 = self.convAndPool(x, self.conv1_0, self.batchnorm1, self.pool, self.relu)
        x0 = self.convAndPool(x0, self.conv1_1, self.batchnorm4, self.pool, self.relu)
        x0 = x0.view(x0.size(0), -1)


        x1 = self.convAndPool(x, self.conv2_0, self.batchnorm2, self.pool, self.relu)
        x1 = self.convAndPool(x1, self.conv2_1, self.batchnorm5, self.pool, self.relu)
        x1 = x1.view(x1.size(0), -1)

        x2 = self.convAndPool(x, self.conv3_0, self.batchnorm3, self.pool, self.relu)
        x2 = self.convAndPool(x2, self.conv3_1, self.batchnorm6, self.pool, self.relu)
        x2 = x2.view(x2.size(0), -1)

        # print(x0.shape, x1.shape, x2.shape)

        x = torch.cat((x0, x1, x2), 1)

        # print(x.shape)

        # x = x.view(x.size(0), -1)

        # (print(x.shape))
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        # x = self.relu(self.fc1(x))
        x = self.softmax(self.fc4(x))
        # x = self.softmax(self.fc2(x))

        # print(x.shape)

        return x

class Classifier_v14(nn.Module):
    def __init__(self):
        super(Classifier_v14, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 7)
        self.conv2 = torch.nn.Conv2d(16, 32, 5)
        # self.conv3 = torch.nn.Conv2d(32, 64, 3, padding='same')
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.softmax = torch.nn.Softmax(dim=1)
        self.fc1 = torch.nn.Linear(32 * 12 * 12, 200)
        self.fc2 = torch.nn.Linear(200, 100)
        self.fc3 = torch.nn.Linear(100, 20)
        self.fc4 = torch.nn.Linear(20, 2)
        self.sigmoid = torch.nn.Sigmoid()
        # self.fc2 = torch.nn.Linear(100, 2)

        self.dropout = torch.nn.Dropout(0.5)
        

        self.batchnorm1 = torch.nn.BatchNorm2d(16)
        self.batchnorm2 = torch.nn.BatchNorm2d(32)
        self.batchnorm3 = torch.nn.BatchNorm2d(64)
        self.batchnorm4 = torch.nn.BatchNorm2d(128)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.batchnorm1(x)
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.batchnorm2(x)
        x = self.pool(x)
        # x = self.relu(self.conv3(x))
        # x = self.batchnorm3(x)
        # x = self.pool(x)
        # x = self.relu(self.conv4(x))
        # x = self.batchnorm4(x)
        # x = self.pool(x)



        # print(x.shape)

        x = x.view(x.size(0), -1)

        # (print(x.shape))
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        # x = self.relu(self.fc1(x))
        x = self.softmax(self.fc4(x))
        # x = self.softmax(self.fc2(x))

        # print(x.shape)

        return x

class Classifier_v13(nn.Module):
    def __init__(self):
        super(Classifier_v13, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 7)
        self.conv2 = torch.nn.Conv2d(16, 32, 5)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, padding='same')
        self.conv4 = torch.nn.Conv2d(64, 128, 3)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.softmax = torch.nn.Softmax(dim=1)
        self.fc1 = torch.nn.Linear(128 * 2 * 2, 200)
        self.fc2 = torch.nn.Linear(200, 100)
        self.fc3 = torch.nn.Linear(100, 20)
        self.fc4 = torch.nn.Linear(20, 2)
        self.sigmoid = torch.nn.Sigmoid()
        # self.fc2 = torch.nn.Linear(100, 2)

        self.dropout = torch.nn.Dropout(0.5)
        

        self.batchnorm1 = torch.nn.BatchNorm2d(16)
        self.batchnorm2 = torch.nn.BatchNorm2d(32)
        self.batchnorm3 = torch.nn.BatchNorm2d(64)
        self.batchnorm4 = torch.nn.BatchNorm2d(128)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.batchnorm1(x)
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.batchnorm2(x)
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.batchnorm3(x)
        x = self.pool(x)
        x = self.relu(self.conv4(x))
        x = self.batchnorm4(x)
        x = self.pool(x)



        # print(x.shape)

        x = x.view(x.size(0), -1)

        # (print(x.shape))
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        # x = self.relu(self.fc1(x))
        x = self.softmax(self.fc4(x))
        # x = self.softmax(self.fc2(x))

        # print(x.shape)

        return x



class Classifier_v12(nn.Module):
    def __init__(self):
        super(Classifier_v12, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 7)
        self.conv2 = torch.nn.Conv2d(16, 32, 5)
        self.conv3 = torch.nn.Conv2d(32, 64, 3)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.softmax = torch.nn.Softmax(dim=1)
        self.fc1 = torch.nn.Linear(64 * 5 * 5, 200)
        self.fc2 = torch.nn.Linear(200, 100)
        self.fc3 = torch.nn.Linear(100, 20)
        self.fc4 = torch.nn.Linear(20, 2)
        self.sigmoid = torch.nn.Sigmoid()
        # self.fc2 = torch.nn.Linear(100, 2)

        self.dropout = torch.nn.Dropout(0.75)
        

        self.batchnorm1 = torch.nn.BatchNorm2d(16)
        self.batchnorm2 = torch.nn.BatchNorm2d(32)
        self.batchnorm3 = torch.nn.BatchNorm2d(64)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.batchnorm1(x)
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.batchnorm2(x)
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.batchnorm3(x)
        x = self.pool(x)


        # print(x.shape)

        x = x.view(x.size(0), -1)

        # (print(x.shape))
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        # x = self.relu(self.fc1(x))
        x = self.softmax(self.fc4(x))
        # x = self.softmax(self.fc2(x))

        # print(x.shape)

        return x

class Classifier_v11(nn.Module):
    def __init__(self):
        super(Classifier_v11, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 7)
        self.conv2 = torch.nn.Conv2d(16, 32, 5)
        self.conv3 = torch.nn.Conv2d(32, 64, 3)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.softmax = torch.nn.Softmax(dim=1)
        self.fc1 = torch.nn.Linear(64 * 5 * 5, 200)
        self.fc2 = torch.nn.Linear(200, 20)
        self.fc3 = torch.nn.Linear(20, 2)
        self.sigmoid = torch.nn.Sigmoid()
        # self.fc2 = torch.nn.Linear(100, 2)

        self.dropout = torch.nn.Dropout(0.75)
        

        self.batchnorm1 = torch.nn.BatchNorm2d(16)
        self.batchnorm2 = torch.nn.BatchNorm2d(32)
        self.batchnorm3 = torch.nn.BatchNorm2d(64)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.batchnorm1(x)
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.batchnorm2(x)
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.batchnorm3(x)
        x = self.pool(x)


        # print(x.shape)

        x = x.view(x.size(0), -1)

        # (print(x.shape))
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        # x = self.relu(self.fc1(x))
        x = self.softmax(self.fc3(x))
        # x = self.softmax(self.fc2(x))

        # print(x.shape)

        return x

class Classifier_v10(nn.Module):
    def __init__(self):
        super(Classifier_v10, self).__init__()
        self.conv1_7 = torch.nn.Conv2d(3, 16, 7,stride=1, padding=2)
        self.conv1_5 = torch.nn.Conv2d(3, 16, 5,stride=1, padding=1)
        self.conv1_3 = torch.nn.Conv2d(3, 16, 3,stride=1)

        self.conv2 = torch.nn.Conv2d(48, 48, 3, stride=1)
        self.conv3 = torch.nn.Conv2d(48, 96, 3, stride=1)

        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.softmax = torch.nn.Softmax(dim=1)

        self.fc1 = torch.nn.Linear(96 * 2 * 2, 20)
        self.fc2 = torch.nn.Linear(20, 2)
        self.sigmoid = torch.nn.Sigmoid()
        # self.fc2 = torch.nn.Linear(100, 2)

        self.dropout = torch.nn.Dropout(0.75)
        

        self.batchnorm1 = torch.nn.BatchNorm2d(16)
        self.batchnorm2 = torch.nn.BatchNorm2d(48)
        self.batchnorm3 = torch.nn.BatchNorm2d(96)


    def convAndPool(self, x, conv, pool, activation, batchnorm):
        x = activation(conv(x))
        x = batchnorm(x)
        x = pool(x)
        return x
    

    def forward(self, x):
        x = self.pool(x)
        x_7 = self.convAndPool(x, self.conv1_7, self.pool, self.relu, self.batchnorm1)
        x_5 = self.convAndPool(x, self.conv1_5, self.pool, self.relu, self.batchnorm1)
        x_3 = self.convAndPool(x, self.conv1_3, self.pool, self.relu, self.batchnorm1)

        # print(x_7.shape, x_5.shape, x_3.shape)
        x = torch.cat((x_7, x_5, x_3), 1)

        x = self.convAndPool(x, self.conv2, self.pool, self.relu, self.batchnorm2)
        x = self.convAndPool(x, self.conv3, self.pool, self.relu, self.batchnorm3)
        # print(x.shape)

        x = x.view(x.size(0), -1)

        # (print(x.shape))
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        # x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        # x = self.softmax(self.fc2(x))

        # print(x.shape)

        return x


class Classifier_v9(nn.Module):
    def __init__(self):
        super(Classifier_v9, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 7)
        self.conv2 = torch.nn.Conv2d(16, 32, 5)
        self.conv3 = torch.nn.Conv2d(32, 64, 3)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.softmax = torch.nn.Softmax(dim=1)
        self.fc1 = torch.nn.Linear(64 * 5 * 5, 20)
        self.fc2 = torch.nn.Linear(20, 2)
        self.sigmoid = torch.nn.Sigmoid()
        # self.fc2 = torch.nn.Linear(100, 2)

        self.dropout = torch.nn.Dropout(0.75)
        

        self.batchnorm1 = torch.nn.BatchNorm2d(16)
        self.batchnorm2 = torch.nn.BatchNorm2d(32)
        self.batchnorm3 = torch.nn.BatchNorm2d(64)


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.batchnorm1(x)
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.batchnorm2(x)
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.batchnorm3(x)
        x = self.pool(x)


        # print(x.shape)

        x = x.view(x.size(0), -1)

        # (print(x.shape))
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        # x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        # x = self.softmax(self.fc2(x))

        # print(x.shape)

        return x

class Classifier_v8(nn.Module):
    def __init__(self):
        super(Classifier_v8, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 128, 7)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(128, 256, 5)

        self.fc1 = torch.nn.Linear(256 * 12 * 12, 100)
        self.fc2 = torch.nn.Linear(100, 2)

        self.convDrop = torch.nn.Dropout(0.1)
        self.dropout = torch.nn.Dropout(0.5)

        self.batchnorm1 = torch.nn.BatchNorm2d(128)
        self.batchnorm2 = torch.nn.BatchNorm2d(256)

    
    def computeUnit(self, x, conv, batchnorm, pool, dropout):
        x = conv(x)
        x = batchnorm(x)
        x = self.relu(x)
        x = pool(x)
        x = dropout(x)
        return x

    def forward(self, x):
        x = self.computeUnit(x, self.conv1, self.batchnorm1, self.pool, self.convDrop)
        x = self.computeUnit(x, self.conv2, self.batchnorm2, self.pool, self.convDrop)

        # print(x.shape)
        # x = x.view(-1, 36 * 2 * 2)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class Classifier_v7(nn.Module):
    def __init__(self):
        super(Classifier_v7, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 9, 3)
        self.conv1_1 = torch.nn.Conv2d(9, 9, 3)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(9, 27, 3)
        self.conv2_1 = torch.nn.Conv2d(27, 27, 3)
        self.conv3 = torch.nn.Conv2d(27, 81, 3)
        self.conv3_1 = torch.nn.Conv2d(81, 81, 3)
        self.conv4 = torch.nn.Conv2d(81, 243, 3)
        # self.conv4_1 = torch.nn.Conv2d(243, 243, 3)
        self.fc1 = torch.nn.Linear(243 * 2 * 2, 100)
        self.fc2 = torch.nn.Linear(100, 2)

        self.convDrop = torch.nn.Dropout(0.1)
        self.dropout = torch.nn.Dropout(0.75)

        self.batchnorm1 = torch.nn.BatchNorm2d(9)
        self.batchnorm2 = torch.nn.BatchNorm2d(27)
        self.batchnorm3 = torch.nn.BatchNorm2d(81)
        self.batchnorm4 = torch.nn.BatchNorm2d(243)
    
    def computeUnit(self, x, conv,secondConv, batchnorm, pool, dropout):
        x = conv(x)
        x = secondConv(x)
        x = batchnorm(x)
        x = self.relu(x)
        x = pool(x)
        x = dropout(x)
        return x

    def forward(self, x):
        x = self.computeUnit(x, self.conv1, self.conv1_1, self.batchnorm1, self.pool, self.convDrop)
        x = self.computeUnit(x, self.conv2, self.conv2_1, self.batchnorm2, self.pool, self.convDrop)
        x = self.computeUnit(x, self.conv3, self.conv3_1, self.batchnorm3, self.pool, self.convDrop)
        # x = self.computeUnit(x, self.conv4, self.conv4_1, self.batchnorm4, self.pool, self.convDrop)
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.relu(x)
        # print(x.shape)
        # x = x.view(-1, 36 * 2 * 2)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class Classifier_v6(nn.Module):
    def __init__(self):
        super(Classifier_v6, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 9, 3)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(9, 27, 3)
        self.conv3 = torch.nn.Conv2d(27, 81, 3)
        self.conv4 = torch.nn.Conv2d(81, 243, 3)
        self.fc1 = torch.nn.Linear(243 * 2 * 2, 100)
        self.fc2 = torch.nn.Linear(100, 2)

        self.convDrop = torch.nn.Dropout(0.1)
        self.dropout = torch.nn.Dropout(0.75)

        self.batchnorm1 = torch.nn.BatchNorm2d(9)
        self.batchnorm2 = torch.nn.BatchNorm2d(27)
        self.batchnorm3 = torch.nn.BatchNorm2d(81)
        self.batchnorm4 = torch.nn.BatchNorm2d(243)
    
    def computeUnit(self, x, conv, batchnorm, pool, dropout):
        x = conv(x)
        x = batchnorm(x)
        x = self.relu(x)
        x = pool(x)
        x = dropout(x)
        return x

    def forward(self, x):
        x = self.computeUnit(x, self.conv1, self.batchnorm1, self.pool, self.convDrop)
        x = self.computeUnit(x, self.conv2, self.batchnorm2, self.pool, self.convDrop)
        x = self.computeUnit(x, self.conv3, self.batchnorm3, self.pool, self.convDrop)
        x = self.computeUnit(x, self.conv4, self.batchnorm4, self.pool, self.convDrop)
        # print(x.shape)
        # x = x.view(-1, 36 * 2 * 2)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class Classifier_v5(nn.Module):
    def __init__(self):
        super(Classifier_v5, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 9, 3)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(9, 27, 3)
        self.conv3 = torch.nn.Conv2d(27, 81, 3)
        self.fc1 = torch.nn.Linear(81 * 2 * 2, 100)
        self.fc2 = torch.nn.Linear(100, 20)
        self.fc3 = torch.nn.Linear(20, 2)

        self.convDrop = torch.nn.Dropout(0.1)
        self.dropout = torch.nn.Dropout(0.75)
        self.batchnorm1 = torch.nn.BatchNorm2d(9)
        self.batchnorm2 = torch.nn.BatchNorm2d(27)
        self.batchnorm3 = torch.nn.BatchNorm2d(81)
    
    def computeUnit(self, x, conv, batchnorm, pool, dropout):
        x = conv(x)
        x = batchnorm(x)
        x = self.relu(x)
        x = pool(x)
        x = dropout(x)
        return x

    def forward(self, x):
        x = self.computeUnit(x, self.conv1, self.batchnorm1, self.pool, self.convDrop)
        x = self.computeUnit(x, self.conv2, self.batchnorm2, self.pool, self.convDrop)
        x = self.computeUnit(x, self.conv3, self.batchnorm3, self.pool, self.convDrop)
        # print(x.shape)
        # x = x.view(-1, 36 * 2 * 2)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

class Classifier_v4(nn.Module):
    def __init__(self):
        super(Classifier_v4, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 9, 3)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(9, 18, 3)
        self.fc1 = torch.nn.Linear(18 * 6 * 6, 100)
        self.fc2 = torch.nn.Linear(100, 20)
        self.fc3 = torch.nn.Linear(20, 2)

        self.convDrop = torch.nn.Dropout(0.1)
        self.dropout = torch.nn.Dropout(0.75)
        self.batchnorm1 = torch.nn.BatchNorm2d(9)
        self.batchnorm2 = torch.nn.BatchNorm2d(18)
    
    def computeUnit(self, x, conv, batchnorm, pool, dropout):
        x = conv(x)
        x = batchnorm(x)
        x = self.relu(x)
        x = pool(x)
        x = dropout(x)
        return x

    def forward(self, x):
        x = self.computeUnit(x, self.conv1, self.batchnorm1, self.pool, self.convDrop)
        x = self.computeUnit(x, self.conv2, self.batchnorm2, self.pool, self.convDrop)

        # print(x.shape)
        # x = x.view(-1, 36 * 2 * 2)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class Classifier_v3(nn.Module):
    def __init__(self):
        super(Classifier_v3, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 9, 3)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(9, 18, 3)
        self.fc1 = torch.nn.Linear(18 * 6 * 6, 10)
        self.fc2 = torch.nn.Linear(10, 2)

        self.convDrop = torch.nn.Dropout(0.1)
        self.dropout = torch.nn.Dropout(0.9)
        self.batchnorm1 = torch.nn.BatchNorm2d(9)
        self.batchnorm2 = torch.nn.BatchNorm2d(18)
    
    def computeUnit(self, x, conv, batchnorm, pool, dropout):
        x = conv(x)
        x = batchnorm(x)
        x = self.relu(x)
        x = pool(x)
        x = dropout(x)
        return x

    def forward(self, x):
        x = self.computeUnit(x, self.conv1, self.batchnorm1, self.pool, self.convDrop)
        x = self.computeUnit(x, self.conv2, self.batchnorm2, self.pool, self.convDrop)

        # print(x.shape)
        # x = x.view(-1, 36 * 2 * 2)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class Classifier_v2(nn.Module):
    def __init__(self):
        super(Classifier_v2, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 9, 3)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(9, 18, 3)
        self.conv3 = torch.nn.Conv2d(18, 36, 3)
        self.fc1 = torch.nn.Linear(36 * 2 * 2, 100)
        self.fc2 = torch.nn.Linear(100, 10)
        self.fc3 = torch.nn.Linear(10, 2)

        self.dropout = torch.nn.Dropout(0.25)
        self.batchnorm1 = torch.nn.BatchNorm2d(9)
        self.batchnorm2 = torch.nn.BatchNorm2d(18)
        self.batchnorm3 = torch.nn.BatchNorm2d(36)


        ## new dropout vals:
        self.drop1 = torch.nn.Dropout(0.3)
        self.drop2 = torch.nn.Dropout(0.5)
        self.drop3 = torch.nn.Dropout(0.6)

        self.dropfc = torch.nn.Dropout(0.75)
    
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

        # x = x.view(-1, 36 * 2 * 2)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropfc(x)
        x = self.relu(self.fc2(x))
        x = self.dropfc(x)
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
    