from realTimeInference import main
import os


rootPath = 'C:/Users/noahv/OneDrive/NDSU Research/Microfluidic ML Device/'

modelpath = rootPath + \
    'Videos/Train Images/3D Train/FullTrainDataset/Weights/01182023_ReduceLRonPlateau_model_2xscale.pt'
imagepath = rootPath + 'Videos/Cancer Cells December 15/25x/DMEM/HPNEnMIA'
saveimagepath = rootPath + 'Videos/Cancer Cells December 15/25x/DMEM/HPNEnMIAOutput/'


videofolder = os.listdir(imagepath)
magnification = 25


if __name__ == '__main__':
    for video in videofolder:
        name = video[:-4]
        print('name', name)
        print('video', video)
        videoPath = imagepath + '/' + video
        newsavefolder = saveimagepath + '/' + name

        if not os.path.exists(newsavefolder):
            os.makedirs(newsavefolder)

        newsavepath = newsavefolder + '/'

        savecsvpath = saveimagepath + name + '.csv'

        print('video path', videoPath)
        print('new save path', newsavefolder)
        print('save csv output', savecsvpath)

        main(videoPath, newsavepath, savecsvpath, modelpath, magnification)
