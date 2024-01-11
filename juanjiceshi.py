import pandas as pd
import math
from keras.layers import *
from keras.models import *
import warnings
from xlutils.copy import copy
import xlrd as xr

num = 2 #1，2，3，4分别表示对应的信号
TEST_MANIFEST_DIR = r'D:\Desktop\two.xls'
TIME_PERIODS = 1000#表示一个组中有多少数据，将29400个数据分为29组，每组1000个
Batch_size = 10#表示一次读入多少个数据
Long = 145#数据中组的个数
Lens = 36#表示验证集的个数

def convert2oneHot(index, lens):
    hot = np.zeros((lens,))
    hot[int(index)] = 1
    return(hot)

#读入测试集，并预处理
def ts_gen(path=TEST_MANIFEST_DIR, batch_size=Batch_size):
    img_list = []
    tablename = ['test1', 'test2', 'test3', 'test4', 'test5', 'test6', 'test7', 'test8', 'test9', 'test10', 'test11',
                 'test12']
    for n in range(12):

        df = pd.read_excel(path, sheet_name=tablename[n], names=['value'], usecols=[num])
        data = df.values
        temp = []
        i = 0
        for j in range(i * 1000, i * 1000 + 1000):
            temp.append(data[j][0])
        temp.append(data[999][0])
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

#加载模型，测试，得出结果，并保存
if __name__ == "__main__":

    test_iter = ts_gen()
    string = "D:\Desktop\data\\best_model.20-1.2187.h5"
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
    oldwb = xr.open_workbook(file)
    newwb = copy(oldwb)
    newws = newwb.get_sheet(num - 1)
    for i in range(len(pres)):
        for j in range(len(pres[0])):
            newws.write(i + 1, j + 1, pres[i][j])
    ohpres = np.argmax(pres, axis=1)
    ohpres = ohpres[:12]
    ohpres = ohpres.tolist()
    for i in range(len(ohpres)):
        newws.write(i + 1, 0, ohpres[i])
    newwb.save(file)  # 保存修改
    print(ohpres[:12])







