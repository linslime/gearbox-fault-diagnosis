import pandas as pd
import numpy as np

path =r'D:\Desktop\two.xls'
img_list = []
tablename=['test1','test2','test3','test4','test5','test6','test7','test8','test9','test10','test11','test12']
for n in range(12):
    pass
    df = pd.read_excel(path, sheet_name=tablename[n], names=['value'], usecols=[1])
    data = df.values
    # print(data)
    temp = []
    i = 0
    for j in range(i * 1000, i * 1000 + 1000):
        temp.append(data[j][0])
    img_list.append(np.array(temp))

np.random.shuffle(img_list)
img_list= np.array(img_list)
img_list= np.array(img_list)
print(len(img_list))
print(len(img_list[1]))
print(img_list)