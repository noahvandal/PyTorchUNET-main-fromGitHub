import cv2
import torch
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
from auxiliary import suppressUndesirableEllipses


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
        length = np.max(axes[0], axes[1])
        xStart = int(x - 0.5*length)
        yStart = (y - 0.5*length)

        segment = image[yStart:(yStart + length), xStart:(xStart + length)] ## ensure axes are right. OpenCV is odd in that color channels are BGR, which makes for an intersting time. 
        segment = cv2.resize(segment, resizeSize, cv2.INTER_LINEAR)

        return segment
    
    # given list of contours, will make ellipses from them; will also send predicted ellipses to classifier network to determine class. 
    def ellipseFromCoordsClassifier(self, contours, RGBimg, classifyNet, classList):
        ell_coord = []
        all_coord = []
        # print(contours)
        for contour in contours:
            if len(contour) >= 5:
                (cx, cy), (lx, ly), angle = cv2.fitEllipseDirect(contour)
                ell_coord = [cx, cy, lx, ly, angle, 1, 0]
                coord = (cx,cy)
                axes = (lx,ly)
                resize = (32, 32) ## default input for the classifcation network. This value will have to be changed if network input size changes. 

                # getting rid of nan and inf values
                for i, var in enumerate(ell_coord):
                    if (np.isnan(ell_coord[i]) == True):
                        ell_coord[i] = 0.01
                    # setting entire-box bounds that are found as non found circles
                    if (np.isinf(ell_coord[i]) == True):
                        ell_coord[i] = 0.01

                segment = self.createImageRegion(RGBimg, coord,axes,resize)   ## segmenting region cell is in. 
                with torch.no_grad():
                    outputs = classifyNet(segment)
                    classPrediction = classifyNet(segment) ## prediction of class from classification network
                    _, index = torch.max(classPrediction)

                    classPrediction = classList(index)

                    #    adding string of classtype at the end after processing for nan values
                    ell_coord.append(classPrediction)
                    # counter to determine what class was recognized most with ID
                    ell_coord.append(1)
                    # print(ell_coord)

            if len(contour) < 5:
                continue

        # will return list of ellipses and their respective coordinates.
        return all_coord

    def ellipseCoords(self, img, classtype):
        # print(mode1(img))
        labels = self.watershedSegment(img)
        contours = self.watershedEllipseFinder(img, labels)
        ellipses = self.ellipseFromCoords(contours, classtype)

        return ellipses
    
    def ellipseCoordsClassifier(self, BWimg, RGBimg, classifyNet, classList):
        labels = self.watershedSegment(BWimg)  ## get center of each region in image
        contours = self.watershedEllipseFinder(BWimg,labels)  ## find contours and fit ellipse to it. 
        ellipses = self.ellipseFromCoordsClassifier(contours, RGBimg, classifyNet, classList)

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

def getEllipsesFromClassListClassifier(BWimg, RGBimg,classifyNet, classList): 
    ellipseCoord = ellipseDetection()

    ellipses = ellipseCoord.ellipseCoordsClassifier(BWimg,RGBimg,classifyNet,classList)  
    ellipses = suppressUndesirableEllipses(ellipses)  ### getting rid of any ellipses that do not fit appropriate parameters. 

    return ellipses

