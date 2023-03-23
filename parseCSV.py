import os
import pandas as pd
import csv
import numpy as np

rootPath = '/home/noahvandal/my_project_dir/my_project_env/UNET_Color/VideoInferences/'

csvPath = rootPath + 'Cancer Cells February 13/1_9 Ratio/Output/'

csvSave = rootPath + 'Cancer Cells February 13/1_9 Ratio/All1_9Ratio.csv'

dirlist = os.listdir(csvPath)
csvoutput= []
csvoutput = np.array(csvoutput)

# df = pd.DataFrame()

def createCSVDF(rootPath, csvList):
    # df_list = pd.DataFrame()
    newdf = pd.DataFrame()
    df1 = pd.DataFrame()
    df = pd.DataFrame()
    for i, file in enumerate(csvList):
        newdf = df1
        if i == 0:
            try:
                df = pd.read_csv(rootPath + file)
                df1 = df
            except: 
                continue
        else:
            try:
                df = pd.read_csv(rootPath + file, skiprows=0)
                df1 = pd.DataFrame(np.vstack([df,newdf]))

            except:
                continue
        
        print(df.shape, newdf.shape)
        # df1 = pd.concat([df.reset_index(drop=True), newdf.reset_index(drop=True)], axis=1, ignore_index=False)
        # df1 = pd.DataFrame(np.vstack([df,newdf]))

    return df1


csvlist = []
for item in dirlist:
    if item[-3:] == 'csv': ## getting only csv files
        csvlist.append(item)
    
print(csvlist)
df = createCSVDF(csvPath, csvlist)

# writer_40 = pd.(rootPath + '5_40_All_CELLWEIGHT.xlsx', engine='xlsxwriter')

df.to_csv(csvSave)

# for item in dirlist:
#     # print(csv[-3:])
#     if item[-3:] == 'csv':
#         # csvlist.append(csv)
#         # print(csvPath + csv)
#         try:
#             csvread = pd.read_csv(csvPath + item)
#             csvread = np.array(csvread)
#             print(csvread.shape, csvoutput.shape)
#             csvoutput = np.vstack((csvread,csvoutput))
#         except:
#             continue
#         print(csvread.shape, len(csvoutput))
#         # csvread = np.array(csvread)
#         # csvoutput = np.vstack([csvread,csvoutput])
#         # csvoutput.extend(csvread)
#         # df.append(csvread)
#         # print(csv)
# # print(csvlist)
# print(csvoutput)

# csvoutput = np.array(csvoutput)



# with open(csvSave, 'w') as csvwrite:
#     writer = csv.writer(csvwrite)
#     writer.writerows(csvoutput)

# print(csvoutput)
