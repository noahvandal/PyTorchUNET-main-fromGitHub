import cv2
import numpy as np
from dataset import rgbToOnehotSparse,rgbToOnehotNew, color2label, cellColor2Label
from collections import namedtuple
import csv
from tracking import Tracking


# inverted dictionary because cv2 reads


def showImageDelay(image, delay, text):
    cv2.imshow(text, image)
    if delay is not None:
        cv2.waitKey(delay)
    else:
        cv2.waitKey(1)
    cv2.destroyAllWindows()


def saveImage(image, imageName, savePath, secondImage=False):
    imageSavePath = savePath + imageName + '.png'
    # print(imageSavePath)

    # print(image.shape, secondImage.shape)

    if secondImage is not False:
        image = scaleImage(image)
        secondImage = scaleImage(secondImage)
        fImage = stackImages(image, secondImage)
        cv2.imwrite(imageSavePath, fImage)

    else:
        image = scaleImage(image)
        cv2.imwrite(imageSavePath, image)


def stackImages(img1, img2):  # both images must be input as numpy array dtype
    # img1 dimensions will be dimensions used
    h, w, _ = img1.shape

    if (img2.shape[0] != h) or (img2.shape[1] != w):
        cv2.resize(img2, (h, w))

    # print('im2', img2.shape)
    # print(img2)

    if len(img2.shape) == 2:
        img2 = np.expand_dims(img2, axis=-1)
        img2 = np.float32(img2)
        # print(img2.shape)
        # print(img2.dtype)
        # if is single channel grayscale, conver to rgb for concatenation and better saving.
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

    outimage = np.hstack([img1, img2])

    # showImage(outimage, 1500, None)

    return outimage


def scaleImage(image):
    maxPixel = np.max(image)
    # print(maxPixel)
    if maxPixel > 1:  # if a pixel value is greater than 1, then it is not normalized and is in range 0-255 most likely
        image = image
    else:
        image = 255*image

    return image


def determineROI(img, mask, isConcat = False, path=True, isPSBead=False):
    if path:
        img = cv2.imread(img)
        mask = cv2.imread(mask)
        if isConcat:
            mask = mask[:,-1280:,:]
    else:
        img = np.array(img)
        mask = np.array(mask)
        if isConcat:
            mask = mask[:,-1280:,:]

    # get color labels to create sparse matrix for easy compare
    # print(img.shape,mask.shape)
    img = rgbToOnehotSparse(img, cellColor2Label)
    mask = rgbToOnehotSparse(mask, cellColor2Label)

    # print(np.sum(img == 0), np.sum(mask==0))
    # print(np.sum(img == 1), np.sum(mask==1))
    # print(np.sum(img == 2), np.sum(mask==2))
    # print(img, mask)
    # print(img.shape, mask.shape)

    HPNEoverlap = np.sum(np.logical_and(img == 0, mask == 0))
    HPNEunion = np.sum(np.logical_or(img == 0, mask == 0)) + \
        1  # +1 to ensure no attempted division by 0

    MIAoverlap = np.sum(np.logical_and(img == 1, mask == 1))
    MIAunion = np.sum(np.logical_or(img == 1, mask == 1)) + \
        1  # +1 to ensure no attempted division by 0

    # BGoverlap = np.sum(np.logical_and(img == 2, mask == 2))
    # BGunion = np.sum(np.logical_or(img == 2, mask == 2)) + \
        # 1  # +1 to ensure no attempted division by 0

    if isPSBead:
        PSoverlap = np.sum(np.logical_and(img == 2, mask == 2))
        PSunion = np.sum(np.logical_or(img == 2, mask == 2)) + \
            1  # +1 to ensure no attempted division by 0

        BGoverlap = np.sum(np.logical_and(img == 3, mask == 3))
        BGunion = np.sum(np.logical_or(img == 3, mask == 3)) + \
            1  # +1 to ensure no attempted division by 0

        PSIoU = PSoverlap/PSunion

        HPNEIoU = HPNEoverlap/HPNEunion
        MIAIoU = MIAoverlap/MIAunion
        PSIoU = PSoverlap/PSunion
        BGIoU = BGoverlap/BGunion

        RoIAcc = (HPNEoverlap + MIAoverlap + PSoverlap + BGoverlap) / \
            (HPNEunion + MIAunion + PSunion + BGunion)

        # print(RoIAcc)

        ClassIoU = [HPNEIoU, MIAIoU, PSIoU, BGIoU]

        # print(ClassIoU)

        return ClassIoU, RoIAcc

    else:
        BGoverlap = np.sum(np.logical_and(img == 2, mask == 2))
        BGunion = np.sum(np.logical_or(img == 2, mask == 2)) + \
            1  # +1 to ensure no attempted division by 0

        HPNEIoU = HPNEoverlap/HPNEunion
        MIAIoU = MIAoverlap/MIAunion
        BGIoU = BGoverlap/BGunion

        RoIAcc = (HPNEoverlap + MIAoverlap + BGoverlap) / \
            (HPNEunion + MIAunion + BGunion)

        # print(RoIAcc)

        ClassIoU = [HPNEIoU, MIAIoU, BGIoU]

        # print(ClassIoU)

        return ClassIoU, RoIAcc

    HPNEIoU = HPNEoverlap/HPNEunion
    MIAIoU = MIAoverlap/MIAunion
    PSIoU = PSoverlap/PSunion
    BGIoU = BGoverlap/BGunion

    RoIAcc = (HPNEoverlap + MIAoverlap + PSoverlap + BGoverlap) / \
        (HPNEunion + MIAunion + PSunion + BGunion)

    # print(RoIAcc)

    ClassIoU = [HPNEIoU, MIAIoU, PSIoU, BGIoU]

    return ClassIoU, RoIAcc


def whatPercentIsClass(imgpath):
    img = cv2.imread(imgpath)
    # get color labels to create sparse matrix for easy compare
    img = rgbToOnehotSparse(img, color2label)

    HPNEpix = np.sum(img == 0)
    MIApix = np.sum(img == 1)
    PSpix = np.sum(img == 3)
    BGpix = np.sum(img == 2)

    imgSize = img.shape[0] * img.shape[1]

    percent = [HPNEpix/imgSize, MIApix/imgSize, PSpix/imgSize, BGpix/imgSize]

    # print(percent)

    return percent


def pasteTextOnImage(image, text, loc=(0.2, 0.1)):
    h, w, _ = image.shape
    image = image.copy()
    image = np.array(image)

    x = int(h * loc[0])
    y = int(w * loc[1])

    image = cv2.putText(image, str(text), (x, y),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    return image


def sparseToRGB(sparse, color_dict):
    onesparsehot = np.array(sparse)

    sparse = np.transpose(sparse, [1, 2, 0])
    output = np.zeros(sparse.shape[0:2]+(3,))
    for i, color in enumerate(color_dict.keys()):
        # print(onehot.shape, output.shape, i, k)
        if i < len(color_dict.keys()):
            output[np.all(sparse == i, axis=2)] = color
    return np.uint8(output)


def showImage(image, n):  # ensure image is inputted as numpy array
    cv2.imshow('Image', image)
    cv2.waitKey(n)
    cv2.destroyAllWindows()


def saveImageOutput(inputImage, outputImage, imgName, savePath, doISave=False, showOutput=False):
    if (type(outputImage) or (type(inputImage)) == 'numpy.ndarray'):  # ensuring are of numpy array dtype
        outputImage = np.array(outputImage)
        inputImage = np.array(inputImage)

    img = stackImages(inputImage, outputImage)  # concatenating in y direction

    if showOutput:
        showImage(img, 1000)  # will show image for 1 s

    if doISave:  # ensure save only happens if desired
        saveImage(img, imgName, savePath)


# put ellipses on an image and return annotated image
def putEllipsesOnImage(image, ellipseDict, magnification):
    # print(ellipseList)
    # print(ellipseDict.keys())
    for key in ellipseDict.keys():
        # print(len(ellipse))
        ellipse = ellipseDict[key]
        # print(ellipse)
        if len(ellipse) == 9:  # in case some erroneous additions were made for specific classes
            # print(ellipse)
            x, y = int(ellipse[0]), int(ellipse[1])
            minor, major = int(ellipse[2]/2), int(ellipse[3]/2)
            angle = int(ellipse[4])

            avgAxes = (minor + major) / 2
            avgAxes = calibrationCorrection(avgAxes, magnification)
            # avgAxes = f'{float(f"{avgAxes:.3g}"):g}'

            image = cv2.ellipse(image, (x, y), (minor, major),
                                angle, 0, 360, (0, 0, 255), 3)
            image = cv2.putText(image, str(
                ellipse[7]), (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
            image = cv2.putText(image, str(float('%.3g' % avgAxes)) + 'um', (x, y+30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
            image = cv2.putText(image, str(key), (x, y-30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
    return image


# will get rid of ellipses that do not meet certain criterion. Use only for initial ellipse detection
def suppressUndesirableEllipses(ellipseList):
    ellipseListOutput = []
    epsilon = 0.0001  # ensure not dividing by 0
    for ellipse in ellipseList:
        if len(ellipse) == 9:  # in case some erroneous additions were made for specific classes; how the ellipse data point are set up
            minor, major = int(ellipse[2]/2), int(ellipse[3]/2)

            eccentricity = minor / (major + epsilon)
            avgAxes = (major + minor) 

            # qualifiers to look for; as configurations change, qualifiers may also change.
            if (eccentricity > 0.3) and (avgAxes > 5):  ## (4/17/23) changed eccentricity to 0.3 forom 0.6 due to causing tracking problems
                ellipseListOutput.append(ellipse)
                # print('i was good!', ellipse)

        if len(ellipse) != 8:
            continue

    return ellipseListOutput


def calibrationCorrection(lengthValues, magnification):
    # see excel worksheet, 'Calibration Data' for information regarding this value
    # adjust this value; accounts for postporcessing (watershed, et.al) techniques that are partially inaccurate
    fudgeFactor = 1.14
    calLength = fudgeFactor * lengthValues/(2.61/(25/magnification))
    return calLength
