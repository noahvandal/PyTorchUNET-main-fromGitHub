import torch
from PIL import Image
import os
import random
import numpy as np
from dataset import color2label

imgSrc = 'C:/Users/noahv/OneDrive/NDSU Research/Microfluidic ML Device/Videos/Train Images/3D Train/TrainDataset/Train/Masks/'


def rgbToOnehotNew(rgb, colorDict):
    print(type(rgb), rgb.shape)
    dense = np.zeros(rgb.shape[:2])
    # print(rgb.shape, dense.shape)
    # print(rgb.dtype, dense.dtype)
    for label, color in enumerate(colorDict.keys()):
        print(label, color)
        # print(type(label), type(color), type(dense), type(rgb))
        # print(rgb.shape, dense.shape, color, label)
        if label < len(colorDict.keys()):
            dense[np.all(rgb == color, axis=-1)] = label
            # pixelSum = np.sum(np.array(rgb) == color)
            # print(pixelSum)

    print(dense)
    return dense


def onehot_to_rgb(onehot, color_dict):
    onehot = np.array(onehot)
    # print(onehot)
    # single_layer = np.argmax(onehot, axis=0)
    # print(single_layer)
    onehot = np.transpose(onehot, [1, 2, 0])
    output = np.zeros(onehot.shape[0:2]+(3,))
    for i, k in enumerate(color_dict.keys()):
        # print(onehot.shape, output.shape, i, k)
        if i < len(color_dict.keys()):
            output[np.all(onehot == i, axis=2)] = k
    return np.uint8(output)


imglist = os.listdir(imgSrc)

imgChoice = random.choice(imglist)

imgPath = imgSrc + imgChoice
img = Image.open(imgPath)

# img = Image.fromarray(img, mode="P")
# img.putpalette([
# 255, 255, 255,   # index 0
# 144, 0, 0,  # index 1
# 0, 255, 0,  # index 2
# 0, 0, 255,  # index 3
# ... and so on, you can take it from here.
# ])
# img.show()
# img.show()
img = np.array(img)
# # print(img.shape)
img = rgbToOnehotNew(img, color2label)

# # print(img.shape)

img = np.expand_dims(img, 0)
# # print(img.shape)
outimg = onehot_to_rgb(img, color2label)
# # print(outimg.shape)

outimg = Image.fromarray(outimg)
outimg.show()
# # for row in img:
# # print(row)
# # print(img)

# # print(len(imglist))
