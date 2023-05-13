import cv2

## saving each output frame of a video as a png
def saveEachFrame(video, savepath, videoname):
    src = cv2.VideoCapture(video)

    iterator = 0
    while (src.isOpened()):
        ret, frame = src.read()

        framename = str(videoname) + str(iterator) + '.png'
        if frame is not None:
            cv2.imwrite(savepath + framename, frame)

        if iterator % 100 == 0:
            print(iterator)

        iterator += 1

    src.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    rootPath = 'C:/Users/noahv/OneDrive/NDSU Research/Microfluidic ML Device/Videos/Train Images/3D Train/FineTuneFullTrain/'

    videoPath = rootPath + 'Videos/10um_5_40_1280_1.mp4'
    savePath = rootPath + 'Frames/'
    videoname = '10um_5_40_1280_1'

    saveEachFrame(videoPath, savePath, videoname)
