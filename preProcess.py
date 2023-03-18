import numpy as np


class preProcess():
    def __init__(self, image):
        self.targetSize = (960, 1280, 3)
        self.image = image

    def preProcessImage(self):  # ensure image is corrct size and value
        print(self.image)
        # img = self.image / 255  # ensure between 0, 1
        img = np.resize(self.image, self.targetSize)
        img = np.transpose(img, [2, 0, 1])
        # img = img.astype(np.double)
        # img = np.reshape(img,img.shape+(1,))
        # adding two dimensions beoore sending to unet
        img = np.reshape(img, (1,)+img.shape)
        return img
