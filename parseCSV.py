import os
import pandas as pd
import csv
import numpy as np

# rootPath = '/home/noahvandal/my_project_dir/my_project_env/UNET_Color/VideoInferences/Cancer Cells February 13/1_9 Ratio/'

rootPath = '/home/noahvandal/my_project_dir/my_project_env/UNET_Color/VideoInferences/DatasetFeb10/MIA/'
# 

csvPath = rootPath + 'Output/17_April/'

csvSave = rootPath + 'Output/Apr17_All_MIA.csv'

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
        csvlist.append(item)
    
print(csvlist)

frame = createCSVDF(csvPath, csvlist)


frame.to_csv(csvSave)
