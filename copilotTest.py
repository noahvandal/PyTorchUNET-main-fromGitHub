import numpy as np
import torch


def convertRGBtoGray(image, isTorch=False):
    image = np.array(image)
    image = np.transpose(image, [1, 2, 0])
    image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])  ##pseudo human eye conversion
    image = np.transpose(image, [2, 0, 1])
    if isTorch:
        image = np.sum(image, axis=-1) ## sum along the channel axis
        image = np.transpose(image, [2, 0, 1])
        image = torch.from_numpy(image)
    else:
        image = np.sum(image, axis=-1) ## sum along the channel axis
    return image