import os
import pandas as pd
import csv
import numpy as np

typeList = ['HPNE/','MIA/','99_1 Ratio/','1_9 Ratio/']
nameTypeList = ['HPNE','MIA','99_1Ratio','1_9Ratio']

numType = 0

# rootPath = '/home/noahvandal/my_project_dir/my_project_env/UNET_Color/VideoInferences/Cancer Cells February 13/' + typeList[numType]

# rootPath = '/home/noahvandal/my_project_dir/my_project_env/UNET_Color/VideoInferences/DatasetFeb10/' + typeList[numType]
# 
rootPath = '/home/noahvandal/my_project_dir/my_project_env/UNET_Color/VideoInferences/DatasetDec15/' + typeList[numType]

csvPath = rootPath + 'Output/11_May/'

csvSave = rootPath + 'Output/11_May' + nameTypeList[numType] + '.csv'

dirlist = os.listdir(csvPath)
csvoutput= []
csvoutput = np.array(csvoutput)

# d

def createCSVDF(rootPath, csvList): ## creatign function to return a dataframe of all csv files in a single file
    frame = pd.DataFrame()
    for i, filename in enumerate(csvList):
        try:
            df = pd.read_csv(rootPath +  filename, index_col=None)
            print(i)
            if i == 1:
                frame = df.copy()
                print(frame.shape, df.shape)
                print('i is 0')
            else:
                print(frame.shape, df.shape)
                print(type(frame), type(df))
                frame = pd.DataFrame(np.vstack([frame, df]))
        except:
            print('skipped!')
        continue

    print('Dataframe created!')
    return frame

csvlist = []
print(dirlist)
for item in dirlist:
    if item[-3:] == 'csv': ## getting only csv files
        if 'acc' not in item: ## removing accuracy files
            csvlist.append(item)
    
print(csvlist)

frame = createCSVDF(csvPath, csvlist)


frame.to_csv(csvSave)
