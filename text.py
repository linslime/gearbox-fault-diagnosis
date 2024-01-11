import pandas as pd
import numpy as np

img_list = []
tablename=['gearbox00','gearbox10','gearbox20','gearbox30','gearbox40']
for n in range(5):
    pass
    df = pd.read_excel(r'D:\Desktop\one.xls', sheet_name=tablename[n], names=['value'], usecols=[1])
    data = df.values
    # print(data)
    for i in range(29):
        temp = []
        for j in range(i * 1000, i * 1000 + 1000 - 1):
            temp.append(data[j][0])
        temp.append(n)
        img_list.append(np.array(temp))
np.random.shuffle(img_list)
img_list= np.array(img_list)
img_list= np.array(img_list)
print(img_list)
print(len(img_list))
print(len(img_list[0]))