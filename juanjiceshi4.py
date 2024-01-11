import keras
from scipy.io import loadmat
import matplotlib.pyplot as plt
import glob
import numpy as np
import pandas as pd
import math
import os
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import warnings
from xlutils.copy import copy
import xlrd as xr


warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)


#这里是我导入的训练集数据训练集，大家对应自己的信号数据就好，数据我下面会发，大家可以看一下数据的格式；

num = 4
#这里是导入的我测试集的数据
TEST_MANIFEST_DIR = r'D:\Desktop\two.xls'
TIME_PERIODS = 899
Batch_size = 10
Long = 145
Lens = 36
def convert2oneHot(index, lens):
    hot = np.zeros((lens,))
    hot[int(index)] = 1
    return(hot)
def ts_gen(path=TEST_MANIFEST_DIR, batch_size=Batch_size):
    img_list = []
    tablename = ['test1', 'test2', 'test3', 'test4', 'test5', 'test6', 'test7', 'test8', 'test9', 'test10', 'test11',
                 'test12']
    for n in range(12):

        df = pd.read_excel(path, sheet_name=tablename[n], names=['value'], usecols=[num])
        data = df.values
        # print(data)
        temp = []
        i = 0
        for j in range(i * 900, i * 900 + 900):
            temp.append(data[j][0])
        temp.append(data[899][0])
        img_list.append(np.array(temp))
    np.random.shuffle(img_list)
    np.random.shuffle(img_list)
    np.random.shuffle(img_list)
    img_list = np.array(img_list)
    img_list = np.array(img_list)
    print(len(img_list))
    print(len(img_list[1]))
    print(img_list)
    img_list = np.array(img_list)[:Lens]
    print("Found %s test items." % len(img_list))
    print("list 1 is", img_list[0, -1])
    steps = math.ceil(len(img_list) / batch_size)
    while True:
        for i in range(steps):
            batch_list = img_list[i * batch_size:i * batch_size + batch_size]
            batch_x = np.array([file for file in batch_list[:, 1:]])
            yield batch_x



if __name__ == "__main__":

    test_iter = ts_gen()
    string = "D:\Desktop\data\\best_model.10-2.3223.h5"
    model = load_model(string, compile=False)
    pres = model.predict_generator(generator=test_iter, steps=math.ceil(528 / Batch_size), verbose=1)
    print(pres.shape)
    ohpres = np.argmax(pres, axis=1)
    print(ohpres.shape)
    df = pd.DataFrame()
    df["id"] = np.arange(1, len(ohpres) + 1)
    df["label"] = ohpres
    df.to_csv("D:\Desktop\data\\predicts.csv", index=None)




tesft_iter = ts_gen()
list = model.predict_generator(generator=test_iter, steps=math.ceil(520 / Batch_size), verbose=1)[:12]
print(list)
temp = []
pres = []
for i in list:
    temp = i.tolist()
    pres.append(temp)
print(pres)
file = "D:\Desktop\gailv.xls"
oldwb = xr.open_workbook(file)  # 打开工作簿
newwb = copy(oldwb)  # 复制出一份新工作簿
newws = newwb.get_sheet(num - 1)  # 获取指定工作表，0表示实际第一张工作表
for i in range(len(pres)):
    for j in range(len(pres[0])):
        newws.write(i + 1, j + 1, pres[i][j])  # 把列表a中的元素逐个写入第一列，0表示实际第1列,i+1表示实际第i+2行
ohpres = np.argmax(pres, axis=1)
ohpres=ohpres[:12]
ohpres = ohpres.tolist()
for i in range(len(ohpres)):
    newws.write(i + 1,0,ohpres[i])
newwb.save(file)  # 保存修改
print(ohpres[:12])







