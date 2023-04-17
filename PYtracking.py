import numpy as np
import math


class Tracking():
    def __init__(self, trackingDict, globalList, currentList, currentID, pixelChange, magnification, frameNumber):
        # how many pixels away an object must be to be considered as same ID; appropriate for 2um/min flow rate at image size 1280x960
        self.distanceThresh = 100
        # number of pixels dimensions must be within to be evaluated as same ID; measure of major/minor axes
        self.dimDiffThresh = 10
        # maximum number of ID's can be separated before considered as same ID
        self.IDSepThresh = 40
        self.globalList = globalList  # list of all id's that have passed through
        self.trackingDict = trackingDict  # working dictionary of current id's
        self.currentList = currentList  # list of ID'd objects in current frame
        self.currentID = currentID   # current iterator value of the ID
        self.magnification = magnification
        self.PixelChange = pixelChange
        self.frame = frameNumber

    # uses regular list formatting to compare points
    # input should be two sets of ellispse coordinate / data
    def arePointsSimilar(self, item1, item2):

        isSimilar = False

        index = item1
        index2 = item2

        distance = math.hypot(
            index[0]-index2[0], index[1]-index2[1])
        dimDiff = np.absolute(
            ((index[2] - index2[2])/2) + ((index[3] - index2[3])/2)/2)
        # IDSep = np.absolute(item1[0]-item2[0]) ## for using dictionary, cannot use this metric

        # print(distance)
        # cannot simply delete entries here, because then changes size of list which will result in entries not being indexed.
        if (distance < self.distanceThresh) and (dimDiff < self.dimDiffThresh):
            isSimilar = True

        return isSimilar, distance

    # make sure points aren't duplicated from frame by frame
    def comparePointsList(self):
        add_count = 0
        remove_from_list = True  # generally set to true.
        # the threshold for how close points can be between frames for single object classification, and the current ID number in use (to prevent overlap))
        tracking_dict_copy = self.trackingDict.copy()
        current_list_copy = self.currentList.copy()
        # current_list_copy = list(filter(None, current_list_copy))
        point_list = []
        # print(current_list)

        avgPositionChangeList = []
        avgPositionChange = 0

        for object_id, pt2 in tracking_dict_copy.items():
            object_exists = False
            for pt in current_list_copy:
                # distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                # Update IDs position
                isSimilar, distance = self.arePointsSimilar(pt, pt2)
                if isSimilar:
                    # tracking_objects2[object_id] = tracking_objects[object_id]
                    classType = pt[7]

                    markov_weight = 0.7  # weight to assign to previous value
                    # weight past values to create markov average
                    minor = (1-markov_weight)*pt2[2] + markov_weight*pt[2]
                    # weight past values to create markov average
                    major = (1-markov_weight)*pt2[3] + markov_weight*pt[3]

                    # finding the average number of pixels each point travels
                    # (numFrame*avgdistance + distance / totalnumFrames)
                    avgDistance = (pt2[5] * pt2[6] + distance) / (pt2[5] + 1)
                    numClassEncounter = pt2[8]

                    if pt[7] != pt2[7]:
                        # decrementing class encounter; essentially takes average class detection type
                        if numClassEncounter > 0:  # ensuring that average of class detection type is inferenced
                            classType = pt2[7]
                            numClassEncounter = numClassEncounter - 1
                        if numClassEncounter <= 0:
                            classType = pt[7]
                            numClassEncounter = 0

                        # print('decrement!')
                    if pt[7] == pt2[7]:
                        numClassEncounter = numClassEncounter + 1
                        # print('increment!', pt[8], pt2[8], numClassEncounter)

                    new_point = [pt[0], pt[1], minor,
                                 major, pt[4], pt2[5], avgDistance, classType, numClassEncounter, pt2[9], pt[9]]

                    avgPositionChangeList.append(distance)

                    # update coordinate/radius data for ID in dictionary
                    self.trackingDict[object_id] = new_point
                    object_exists = True
                    # addding point to denote that it was detected as ID'ed from current list
                    point_list.append(pt)
                    if pt in self.currentList:
                        self.currentList.remove(pt)
                    continue

            # Remove IDs lost and append to global dictionary
            if not object_exists:  # if this is set to True, then will remove elements from list if they are not noted in current frame. if set to false, those points will be kept.
                if remove_from_list:

                    id_val = self.trackingDict[object_id]

                    minor = int(id_val[2]/2)
                    major = int(id_val[3]/2)
                    avgDiam = (major + minor) / 2
                    avgDiam = self.calibrationCorrection(avgDiam)
                    Diameter = avgDiam  # I was confused since the fitEllipse function is different than cv2.ellipse in terms of major / minor axes
                    numberFrames = id_val[5]
                    avgPositionChange = id_val[6]

                    coord = (id_val[0], id_val[1])
                    axes = (id_val[2], id_val[3])
                    numFrames = id_val[5]
                    avgSpeed = id_val[6]
                    Identity = id_val[7]
                    numClassEncounter = id_val[8]

                    combine_id_data = [object_id, coord, axes,
                                       numFrames, avgSpeed, Identity, numClassEncounter, Diameter]
                    # making sure to ignore small artifacts
                    if (avgDiam >= 3) and (avgDiam <= 30) and (numberFrames >= 3) and ((4*avgPositionChange) >= self.PixelChange):
                        # storing data regarding beads that have left the frame.
                        self.globalList.append(combine_id_data)
                    # getting rid of unrecognized point from dictionary
                    self.trackingDict.pop(object_id)
                    print('i popped:', object_id)

        for ID in self.trackingDict:  # updating number of frames for each dictionary addition
            point = self.trackingDict[ID]
            # increment the number of frames that has been completed for each point in tracking dictionary
            numFrames = point[5] + 1
            # print(point)
            point = [point[0], point[1], point[2],
                     point[3], point[4], numFrames, point[6], point[7], point[8]]
            self.trackingDict[ID] = point

        for pt in self.currentList:  # adding points that weren't in dicitonary to dictionary
            add_count += 1
            self.trackingDict[self.currentID] = pt
            self.currentID += 1

        if len(current_list_copy) == 0:
            self.trackingDict.clear()  # just in case there are stored keys
#             current_list = []
        # print(tracking_dict)

        # finding average flow rate
        if (len(avgPositionChangeList) != 0):
            avgPositionChange = sum(avgPositionChangeList) / \
                len(avgPositionChangeList)
        if (len(avgPositionChangeList) == 0):
            avgPositionChange = 0

        return self.trackingDict, self.currentID, self.globalList, current_list_copy, avgPositionChange

    def calibrationCorrection(self, lengthValues):
        # see excel worksheet, 'Calibration Data' for information regarding this value
        fudgeFactor = 1.14  # for abnormal pixel values due to post processing
        calLength = fudgeFactor * lengthValues/(2.61/(25/self.magnification))
        return calLength


class postVideoProcessList():
    def __init__(self, globalList):
        self.globalList = globalList
        self.distanceThresh = 100
        # number of pixels dimensions must be within to be evaluated as same ID; measure of major/minor axes
        self.dimDiffThresh = 10

        # self.arepointssimilar = self.tracker.arePointsSimilar
    def arePointsSimilar(self, item1, item2):

        isSimilar = False

        index = item1
        index2 = item2

        distance = math.hypot(
            index[0]-index2[0], index[1]-index2[1])
        dimDiff = np.absolute(
            ((index[2] - index2[2])/2) + ((index[3] - index2[3])/2)/2)
        # IDSep = np.absolute(item1[0]-item2[0]) ## for using dictionary, cannot use this metric

        # print(distance)
        # cannot simply delete entries here, because then changes size of list which will result in entries not being indexed.
        if (distance < self.distanceThresh) and (dimDiff < self.dimDiffThresh):
            isSimilar = True

        return isSimilar

    def SingleComparePointsGlobal(self):
        tempList = []
        copyGlobalList = self.globalList.copy()
        tempCopyGlobal = []
        itemRemoved = False

        for index in self.globalList:
            itemRemoved = False

            coord = index[1]
            axes = index[2]
            # necessary to parse global list format
            item = [coord[0], coord[1], axes[0], axes[1]]

            for index2 in copyGlobalList:
                coord2 = index2[1]
                axes2 = index2[2]
                # necessary to parse global list format
                item2 = [coord2[0], coord2[1], axes2[0], axes2[1]]

                similar = False
                similar = self.arePointsSimilar(item, item2)

                if similar:
                    itemRemoved = True
                if not similar:
                    tempCopyGlobal.append(index2)

            if itemRemoved:  # if the item is found in current working list
                tempList.append(index)

            copyGlobalList = tempCopyGlobal.copy()
            tempCopyGlobal = []

        return tempList

    def comparePointsGlobal(self):
        iterations = 10  # multiple iterations to ensure list is properly parsed

        for i in range(0, iterations):
            # print(len(globalList))
            globalList = self.SingleComparePointsGlobal(globalList)
            print('List match iteration: ', str(i))

        return globalList
