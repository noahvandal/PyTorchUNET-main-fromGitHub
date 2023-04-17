from PIL import Image, ImageDraw
import os

images = []

# rootPath = 'C:/Users/noahv/OneDrive/NDSU Research/Microfluidic ML Device/Videos/August 2022/VideosAllTogether/20um/'
rootPath = '/home/noahvandal/my_project_dir/my_project_env/UNET_Color/VideoInferences/DatasetFeb10/MIA/Output/17_April/'


ImagePath = rootPath + '230203124209041023_2c_v12_pureTest_Linux_drop5_noAugsSingleFolder_t3_bwSeg/'

saveImagePath = rootPath + 'GifOutput/230203124209041023_2c_v12_pureTest_Linux_drop5_noAugsSingleFolder_t3_bwSeg.gif'

imageList = os.listdir(ImagePath)

targetHt = 100


for i, image in enumerate(imageList):
    name = str(i + 1)
    im = Image.open(ImagePath + name + '_All.png')
    width, height = im.size  # width, height
    sizeFactor = targetHt / height
    # print(sizeFactor, width, height)
    im.resize((int(sizeFactor * width), int(sizeFactor * height)))
    images.append(im)

    if i % 20 == 0:
        print(i)

print('Images loaded!')

images[0].save(saveImagePath, format="GIF",
               append_images=images, save_all=True, duration=50, loop=0)
