import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class UNET(nn.Module):
    def __init__(self, in_channels=3, classes=3):
        super(UNET, self).__init__()  # super will inherit parent attributes

        scaleFactor = 2  # this scale factor is what is used to determine size of model; it adjusts number of filters in each layer. 2 is what I used for the model 

        self.layers = [in_channels, int(scaleFactor*8), int(scaleFactor*16), int(
            scaleFactor*32), int(scaleFactor*64), int(scaleFactor*128)]

        self.doubleConvDown = nn.ModuleList([self.__double_conv(
            layer, layerN) for layer, layerN in zip(self.layers[:-1], self.layers[1:])])
        self.upTrans = nn.ModuleList([nn.ConvTranspose2d(layer, layerN, kernel_size=2, stride=2)
                                     for layer, layerN in zip(self.layers[::-1][:-2], self.layers[::-1][1:-1])])
        self.doubleConvUp = nn.ModuleList(
            [self.__double_conv(layer, layer//2) for layer in self.layers[::-1][:-2]])
        self.maxPool2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.finalConv = nn.Conv2d(
            self.layers[1], out_channels=classes, kernel_size=1)

    def __double_conv(self, in_channels, out_channels):
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return conv

    def forward(self, x):
        concatLayers = []
        for down in self.doubleConvDown:
            x = down(x)
            if down != self.doubleConvDown[-1]:
                concatLayers.append(x)
                x = self.maxPool2x2(x)
        concatLayers = concatLayers[::-1]

        for upTrans, doubleConvUp, concatLayer in zip(self.upTrans, self.doubleConvUp, concatLayers):
            x = upTrans(x)
            if x.shape != concatLayer.shape:
                ## reshape x to match concatLayer shape for upsampling
                x = TF.resize(x, concatLayer.shape[2:])

            concatenate = torch.cat((concatLayer, x), dim=1)
            x = doubleConvUp(concatenate)

        x = self.finalConv(x)

        return x
