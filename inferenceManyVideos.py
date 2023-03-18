#%%
from realTimeInference import main
import os


rootPath = '/home/noahvandal/my_project_dir/my_project_env/UNET_Color/'

modelpath = rootPath + 'UNET_MC_PyTorch/FineTuneMarchModel/030323_2x_3c_MainTrain_model.pt'

imagepath = rootPath + 'VideoInferences/Cancer Cells February 13/9_1 Ratio/'

saveimagepath = rootPath + 'VideoInferences/Cancer Cells February 13/Output/9_1Ratio/'


itemlist = os.listdir(imagepath)
videofolder = []

#%%
## parsing through list and seeing if the files are videos or not. 
for item in itemlist:
    if item[-4:] == '.mp4':
        videofolder.append(item)
print(videofolder)

#%%
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

# %%
