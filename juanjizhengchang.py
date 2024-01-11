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


def xs_gen(path=MANIFEST_DIR, batch_size=Batch_size, train=True, Lens=Lens):
    img_list = []
    tablename = ['gearbox00', 'gearbox10', 'gearbox20', 'gearbox30', 'gearbox40']
    for n in range(5):

        df = pd.read_excel(path, sheet_name=tablename[n], names=['value'], usecols=[1])
        data = df.values
        # print(data)
        for i in range(29):
            temp = []

            for j in range(i * 1000, i * 1000 + 1000 + 1):
                temp.append(data[j][0])
            if n==1:
                temp.append(0)
            else:
                temp.append(1)

            img_list.append(np.array(temp))
    np.random.shuffle(img_list)
    np.random.shuffle(img_list)
    np.random.shuffle(img_list)
    np.random.shuffle(img_list)


    if train:
        img_list = np.array(img_list)[:Lens]
        print("Found %s train items." % len(img_list))
        print("list 1 is", img_list[0, -1])
        steps = math.ceil(len(img_list) / batch_size)
    else:
        img_list = np.array(img_list)[Lens:]
        print("Found %s test items." % len(img_list))
        print("list 1 is", img_list[0, -1])
        steps = math.ceil(len(img_list) / batch_size)
    while True:
        for i in range(steps):
            batch_list = img_list[i * batch_size: i * batch_size + batch_size]
            np.random.shuffle(batch_list)
            batch_x = np.array([file for file in batch_list[:, 1:-1]])
            batch_y = np.array([convert2oneHot(label, 2) for label in batch_list[:, -1]])
            yield batch_x, batch_y




def build_model(input_shape=(TIME_PERIODS,), num_classes=2):
    model = Sequential()
    model.add(Reshape((TIME_PERIODS, 1), input_shape=input_shape))

    model.add(Conv1D(16, 8, strides=1, activation='tanh', input_shape=(TIME_PERIODS, 1)))
    model.add(Conv1D(16, 8, strides=1, activation='tanh', padding="same"))
    model.add(MaxPooling1D(2))

    model.add(Conv1D(64, 4, strides=1, activation='tanh', padding="same"))
    model.add(Conv1D(64, 4, strides=1, activation='tanh', padding="same"))
    model.add(MaxPooling1D(2))

    model.add(Conv1D(256, 4, strides=1, activation='tanh', padding="same"))
    model.add(Conv1D(256, 4, strides=1, activation='tanh', padding="same"))
    model.add(MaxPooling1D(2))

    model.add(Conv1D(512, 2, strides=1, activation='tanh', padding="same"))
    model.add(Conv1D(512, 2, strides=1, activation='tanh', padding="same"))
    model.add(MaxPooling1D(2))

    # model.add(Flatten())
    # model.add(Dropout(0.3))
    # model.add(Dense(256, activation='relu'))

    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.9))
    model.add(Dense(num_classes, activation='softmax'))
    return(model)



if __name__ == "__main__":
    train_iter = xs_gen()
    val_iter = xs_gen(train=False)

    ckpt = ModelCheckpoint(
        filepath='D:\Desktop\data\\best_model.{epoch:02d}-{val_loss:.4f}.h5',
        monitor='val_loss', save_best_only=False, verbose=1
    )

    model = build_model()
    opt = keras.optimizers.Adam(0.0002)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])
    print(model.summary())

    train_history = model.fit_generator(
        generator=train_iter,
        steps_per_epoch=Lens // Batch_size,
        epochs=25,
        initial_epoch=0,
        validation_data=val_iter,
        validation_steps=(Long - Lens) // Batch_size,
        callbacks=[ckpt],
    )

    model.save("D:\Desktop\data\\finishModel.h5")




def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.ylabel('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

show_train_history(train_history, 'acc', 'val_acc')

show_train_history(train_history, 'loss', 'val_loss')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(6, 4))
plt.plot(train_history.history['acc'], "g--", label="训练集准确率")
plt.plot(train_history.history['val_acc'], "g", label="验证集准确率")
plt.plot(train_history.history['loss'], "r--", label="训练集损失函数")
plt.plot(train_history.history['val_loss'], "r", label="验证集损失函数")
plt.title('模型的准确率和损失函数', fontsize=14)
plt.ylabel('准确率和损失函数', fontsize=12)
plt.xlabel('世代数', fontsize=12)
plt.ylim(0)
plt.legend()
plt.show()

