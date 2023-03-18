import os
import pandas as pd
import csv
import numpy as np

rootPath = '/home/noahvandal/my_project_dir/my_project_env/UNET_Color/'

csvPath = rootPath + 'DatasetFeb10/MIA/'

csvSave = rootPath + 'DatasetFeb10/MIA/AllHPNE.csv'

dirlist = os.listdir(csvPath)
csvoutput= []

df = pd.DataFrame()

for csv in dirlist:
    # print(csv[-3:])
    if csv[-3:] == 'csv':
        # csvlist.append(csv)
        # print(csvPath + csv)
        try:
            csvread = pd.read_csv(csvPath + csv)
        except:
            continue
        # print(csvread)
        csvoutput.append(csvread)
        # print(csv)
# print(csvlist)
print(csvoutput)

csvoutput = np.array(csvoutput)



with open(csvSave, 'w') as csvwrite:
    writer = csv.writer(csvwrite)
    writer.writerows(csvoutput)

print(csvoutput)
