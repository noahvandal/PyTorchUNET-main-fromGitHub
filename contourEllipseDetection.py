import cv2
import torch
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
from auxiliary import suppressUndesirableEllipses, showImageDelay
# %matplotlib inline ## for displaying in notebook
import matplotlib.pyplot as plt


# this function takes a one-hot output and creates B/W images for each class from the output; input should be (bs,h,w,c) or (h,w,c)
def processOneHotIntoClassBW(img):
    img = np.array(img)  # ensure coorect type
    imgShape = img.shape

    imgList = []  # stores images for each class

    if len(imgShape) == 4:
        img = img[0, :, :, :]

    numClass = img.shape[0]
    for i in range(0, numClass):
        classImg = img[i, :, :]

        # convert object of desire to b/w mask
        classImg[np.all(classImg == i)] = 0
        classImg[np.all(classImg != i)] = 255

        imgList.append(classImg)

    return imgList

## this function takes cooordinates andcontorus as input and outputs a region with the backgroudn zeroed out. 
def negateBackground(feature, image):
    coord, contour = feature
    x, y, w, h = coord

    ## create bw mask of feature
    mask = np.zeros(image.shape, dtype=np.uint8)
    output = np.zeros(image.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, (255,255,255), -1)

    output = np.where(mask > 0, image, mask)

    segment = output[y:(y+h),x:(x+w),:]

    ## how much of image is good content
    zeroPixels = np.sum(np.all(segment == [0,0,0], axis=-1))
    # print(segment.shape)
    totalPixels = segment.shape[0] * segment.shape[1] 
    # print(zeroPixels, totalPixels)

    percentInfo = 1 - zeroPixels / totalPixels # percent of pixels that are not black / total number of pixels


    return segment, percentInfo


class ellipseDetection():

    def watershedSegment(self, img):  # find center of blobs using watershed method
        thresh = img.astype('float32')
        thresh = cv2.threshold(thresh, 100, 255, cv2.THRESH_BINARY)[
            1]  # binarize image
        # find the distance between individual white pixel and nearest black pixel
        distance_map = ndimage.distance_transform_edt(thresh)
        # find centers of features
        thresh = thresh.astype('int32')
        # print(distance_map, thresh)
        local_max = peak_local_max(
            distance_map, indices=False, min_distance=10, labels=thresh)
        markers = ndimage.label(local_max, structure=np.ones((3, 3)))[
            0]  # place labels on max feature image
        # segment image, with markers serving as centers
        labels = watershed(-distance_map, markers, mask=thresh)
        labels = np.array(labels, dtype='uint8')
        return labels  # will return center coordinate of blobs

    # find ellipses from label centers as given
    def watershedEllipseFinder(self, img, labels):
        contour_list = []
        for label in np.unique(labels):
            if label == 0:  # ignoring zero values
                continue
            mask = np.zeros(img.shape, dtype="uint8")
            # changing the mask to either b/w dependent on whether item is present
            mask[labels == label] = 255 
            cnts = cv2.findContours(
                mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # takes first output from 'find contours' (is a tuple)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            c = max(cnts, key=cv2.contourArea)
            contour_list.append(c)

        return contour_list  # returns list of all contours found

    # given list of contours, will make ellipses from them
    def ellipseFromCoords(self, contours, classtype):
        ell_coord = []
        all_coord = []
        # print(contours)
        for contour in contours:
            if len(contour) >= 5:
                (cx, cy), (lx, ly), angle = cv2.fitEllipseDirect(contour)
                ell_coord = [cx, cy, lx, ly, angle, 1, 0]

                # getting rid of nan and inf values
                for i, var in enumerate(ell_coord):
                    if (np.isnan(ell_coord[i]) == True):
                        ell_coord[i] = 0.01
                    # setting entire-box bounds that are found as non found circles
                    if (np.isinf(ell_coord[i]) == True):
                        ell_coord[i] = 0.01
            if len(contour) < 5:
                continue
            # adding string of classtype at the end after processing for nan values
            ell_coord.append(classtype)
            # counter to determine what class was recognized most with ID
            ell_coord.append(1)
            # print(ell_coord)
            all_coord.append(ell_coord)

        # will return list of ellipses and their respective coordinates.
        return all_coord

    def createImageRegion(self, image, coord, axes, resizeSize): ## given an image, central coordinates and axes, output the respective slice of image; resize to desired shape
        x, y = coord

        imH, imW, channels = image.shape ## getting shape to know bounds. 


        minor, major = axes[0], axes[1]

        if (np.isnan(minor) == True):
            minor = 0.01
        # setting entire-box bounds that are found as non found circles
        if (np.isinf(minor) == True):
            minor = 0.01

        if (np.isnan(major) == True):
            major = 0.01
        # setting entire-box bounds that are found as non found circles
        if (np.isinf(major) == True):
            major = 0.01
        

        if(np.isinf(x) == True):
            x = 1
        if(np.isnan(x) == True):
            x = 1
        if(np.isinf(y) == True):
            y = 1
        if(np.isnan(y) == True):
            y = 1

        length = int(np.maximum(major, minor)) ## ensuring thatlength is at least one.
        if length ==0:
            length = 1
        
        # tenpercentlength = int(length / 5) + 1
        # length = length + tenpercentlength   ## just in case segmentation undershoots. 

  
        # print(x, length)
        # print(y,length)
        xStart = int(x - 0.5*length)
        yStart = int(y - 0.5*length)
        xEnd = xStart + length
        yEnd = yStart + length

        ## for catching excepptions where region is out of image bounds. 
        if xStart < 0:
            xStart = 0
            xEnd = xStart + length
        if yStart < 0:
            yStart = 0
            yEnd = yStart + length
        if xEnd > imW:
            xEnd = imW
            xStart = xEnd - length
        if yEnd > imH:
            yEnd = imH
            yStart = yEnd - length

    

        # print(x,y,axes[0],axes[1], length)
        # print(xStart, yStart,xEnd, yEnd)
        h = yEnd - yStart
        w = xEnd - xStart

        coords = xStart, yStart, w, h

        # print(image.shape)
        # testsegment = image[975:995,23:43,:]
        segment = image[yStart:yEnd,xStart:xEnd,:] ## ensure axes are right. OpenCV is odd in that color channels are BGR, which makes for an intersting time. 
        # print(segment.shape,resizeSize)
        segment = cv2.resize(segment, resizeSize, cv2.INTER_LINEAR)
        # print(segment.shape)

        return segment, coords
    
    # given list of contours, will make ellipses from them; will also send predicted ellipses to classifier network to determine class. 
    def ellipseFromCoordsClassifier(self, contours, RGBimg, classifyNet, classList, frameNumber):
        ell_coord = []
        all_coord = []

        imageOutput = []

        # print(contours)
        for contour in contours:
            if len(contour) >= 5:
                (cx, cy), (lx, ly), angle = cv2.fitEllipseDirect(contour)
                ell_coord = [cx, cy, lx, ly, angle, 1, 0]
                coord = (cx,cy)
                axes = (lx,ly)
                resize = (64, 64) ## default input for the classifcation network. This value will have to be changed if network input size changes. 

                # getting rid of nan and inf values
                for i, var in enumerate(ell_coord):
                    if (np.isnan(ell_coord[i]) == True):
                        ell_coord[i] = 0.01
                    # setting entire-box bounds that are found as non found circles
                    if (np.isinf(ell_coord[i]) == True):
                        ell_coord[i] = 0.01
    
                segment, coord = self.createImageRegion(RGBimg, coord,axes,resize)   ## segmenting region cell is in. 

                # print(coord)
                # print(contour)
                blackBackSegment, __ = negateBackground([coord,contour],RGBimg)
                print(type(blackBackSegment))
                print(blackBackSegment.shape)
                blackBackSegment = cv2.resize(blackBackSegment, resize, cv2.INTER_LINEAR)
                # h = cy - 0.5*ly
                # w = cx - 0.5*lx
                testsegment = np.array(segment)
                segment = blackBackSegment
                # imageOutput.append(testsegment)
                # plt.imshow(testsegment, interpolation='nearest')
                # plt.show()
                # showImageDelay(testsegment, 1,'semgent')
                with torch.no_grad():
                    # print(segment.shape)
                    segment = np.transpose(segment, [2, 0, 1])
                    segment = np.expand_dims(segment, 0)
                    # print(segment.shape)
                    segment = torch.from_numpy(segment)
                    segment = segment.float() ## necessary for weight multiplication. 

                    classPrediction = classifyNet(segment) ## prediction of class from classification network
                    # print(classPrediction)
                    _, index = torch.max(classPrediction, dim=1) ## getting largest value for prediction output

                    outputPrediction = np.array(classPrediction)
                    # print(outputPrediction)
                    outputPrediction = [float(outputPrediction[0][0]), float(outputPrediction[0][1])]
                    classPrediction = classList[index]

                    testsegment = cv2.putText(testsegment, classPrediction, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 1, cv2.LINE_AA)
                    imageOutput.append(testsegment)

                    #    adding string of classtype at the end after processing for nan values
                    ell_coord.append(classPrediction)
                    # counter to determine what class was recognized most with ID
                    ell_coord.append(1)
                    ## add on what frame number this ellipse was found in.
                    ell_coord.append(frameNumber)
                    ## placeholder to increment each time ellipse is found in a frame.
                    ell_coord.append(frameNumber)
                    ## the output probabilities
                    ell_coord.append(outputPrediction)
                    # print(ell_coord)

            if len(contour) < 5:
                continue

            all_coord.append(ell_coord)

        # will return list of ellipses and their respective coordinates.
        return all_coord, imageOutput

    def ellipseCoords(self, img, classtype):
        # print(mode1(img))
        labels = self.watershedSegment(img)
        contours = self.watershedEllipseFinder(img, labels)
        ellipses = self.ellipseFromCoords(contours, classtype)

        return ellipses
    
    def ellipseCoordsClassifier(self, BWimg, RGBimg, classifyNet, classList, frameNumber):
        labels = self.watershedSegment(BWimg)  ## get center of each region in image
        contours = self.watershedEllipseFinder(BWimg,labels)  ## find contours and fit ellipse to it. 
        ellipses = self.ellipseFromCoordsClassifier(contours, RGBimg, classifyNet, classList, frameNumber)

        return ellipses




def mode1(x):
    values, counts = np.unique(x, return_counts=True)
    m = counts.argmax()
    return values[m], counts[m]


def getEllipsesFromClassList(imageList, nameList):
    ellipseCoord = ellipseDetection()
    totalEllipses = []  # number of ellipses from each class present
    for i in range(0, len(imageList)-1):
        imgs = imageList[i]

        # passing classlist to associate coordinate with class
        ellipses = ellipseCoord.ellipseCoords(imgs, nameList[i])

        # get rid of ellipses that are not properly ellipses
        ellipses = suppressUndesirableEllipses(ellipses)
        totalEllipses.extend(ellipses)

    return totalEllipses
    


'''
this function also has the input of the rgb source image to segment the region the cells are contained in, for proper classification. 
'''

def getEllipsesFromClassListClassifier(BWimg, RGBimg,classifyNet, classList, frameNumber): 
    ellipseCoord = ellipseDetection()

    # print(BWimg.shape)
    ellipses, imageList = ellipseCoord.ellipseCoordsClassifier(BWimg,RGBimg,classifyNet,classList, frameNumber)  
    ellipses = suppressUndesirableEllipses(ellipses)  ### getting rid of any ellipses that do not fit appropriate parameters. 

    return ellipses, imageList

