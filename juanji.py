import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.layers import *
from keras.models import *
import warnings

warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)


#这里是我导入的训练集数据训练集，大家对应自己的信号数据就好，数据我下面会发，大家可以看一下数据的格式；
MANIFEST_DIR = r'D:\Desktop\one.xls'

#这里是导入的我测试集的数据
TEST_MANIFEST_DIR = r'D:\Desktop\two.xls'
TIME_PERIODS = 1000
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
        pass
        df = pd.read_excel(path, sheet_name=tablename[n], names=['value'], usecols=[1])
        data = df.values
        # print(data)
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



def build_model(input_shape=(TIME_PERIODS,), num_classes=5):
    model = Sequential()
    model.add(Reshape((TIME_PERIODS, 1), input_shape=input_shape))

    model.add(Conv1D(16, 8, strides=2, activation='tanh', input_shape=(TIME_PERIODS, 1)))
    model.add(Conv1D(16, 8, strides=2, activation='tanh', padding="same"))
    model.add(MaxPooling1D(2))

    model.add(Conv1D(64, 4, strides=2, activation='tanh', padding="same"))
    model.add(Conv1D(64, 4, strides=2, activation='tanh', padding="same"))
    model.add(MaxPooling1D(2))

    model.add(Conv1D(256, 4, strides=2, activation='tanh', padding="same"))
    model.add(Conv1D(256, 4, strides=2, activation='tanh', padding="same"))
    model.add(MaxPooling1D(2))

    model.add(Conv1D(512, 2, strides=2, activation='tanh', padding="same"))
    model.add(Conv1D(512, 2, strides=2, activation='tanh', padding="same"))
    model.add(MaxPooling1D(2))

    # model.add(Flatten())
    # model.add(Dropout(0.3))
    # model.add(Dense(256, activation='relu'))

    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    return(model)



if __name__ == "__main__":
    test_iter = ts_gen()
    string = "D:\Desktop\data\\best_model.25-0.8109.h5"
    model = load_model(string, compile=False)
    pres = model.predict_generator(generator=test_iter, steps=math.ceil(528 / Batch_size), verbose=1)
    print(pres.shape)
    ohpres = np.argmax(pres, axis=1)
    print(ohpres.shape)
    df = pd.DataFrame()
    df["id"] = np.arange(1, len(ohpres) + 1)
    df["label"] = ohpres
    df.to_csv("D:\Desktop\data\\predicts.csv", index=None)
    test_iter = ts_gen()
    for x in test_iter:
        x1 = x[0]
        break
    plt.plot(x1)
    plt.show()





plt.show()

tesft_iter = ts_gen()
pres = model.predict_generator(generator=test_iter, steps=math.ceil(520 / Batch_size), verbose=1)
print("ha")
print(pres)
print(len(pres))
print("ha")
# print(ndarray.shape)

ohpres = np.argmax(pres, axis=1)
print(ohpres.shape)
ohpres=ohpres[:12]
print(ohpres)



