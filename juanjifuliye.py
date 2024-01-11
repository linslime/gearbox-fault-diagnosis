import keras
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.layers import *
from keras.models import *
from keras.callbacks import ModelCheckpoint
import warnings
from scipy.fftpack import fft, ifft


warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)


MANIFEST_DIR = r'D:\Desktop\one.xls'#训练集与验证集地址
TIME_PERIODS = 1000#表示一个组中有多少数据，将29400个数据分为29组，每组1000个
Batch_size = 10#表示一次读入多少个数据
Long = 145#数据中组的个数
Lens = 36#表示验证集的个数

def convert2oneHot(index, lens):
    hot = np.zeros((lens,))
    hot[int(index)] = 1
    return(hot)

#读入训练集与验证集，进行傅里叶转换，并预处理
def xs_gen(path=MANIFEST_DIR, batch_size=Batch_size, train=True, Lens=Lens):
    img_list = []
    tablename = ['gearbox00', 'gearbox10', 'gearbox20', 'gearbox30', 'gearbox40']
    for n in range(5):

        df = pd.read_excel(path, sheet_name=tablename[n], names=['value'], usecols=[2])
        data = df.values
        # print(data)
        for i in range(29):
            temp = []

            for j in range(i * 1000, i * 1000 + 1000 + 2):
                temp.append(data[j][0])

            y = temp

            fft_y = fft(y)

            N = len(y)
            x = np.arange(N)
            half_x = x[range(int(N / 2))]

            abs_y = np.abs(fft_y)
            angle_y = np.angle(fft_y)
            normalization_y = abs_y / N
            normalization_half_y = normalization_y[range(int(N / 2))]
            temp = []
            for j in normalization_half_y:
                temp.append(j)
            temp.append(n)
            img_list.append(np.array(temp))

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
            batch_y = np.array([convert2oneHot(label, 5) for label in batch_list[:, -1]])
            yield batch_x, batch_y



#卷积模型神经网络
def build_model(input_shape=(TIME_PERIODS,), num_classes=5):
    model = Sequential()
    model.add(Reshape((TIME_PERIODS, 1), input_shape=input_shape))

    model.add(Conv1D(128, 8, strides=1, activation='tanh', input_shape=(TIME_PERIODS, 1)))
    model.add(Conv1D(128, 8, strides=1, activation='tanh', padding="same"))
    model.add(MaxPooling1D(2))

    model.add(Conv1D(256, 8, strides=1, activation='tanh', padding="same"))
    model.add(Conv1D(256, 8, strides=1, activation='tanh', padding="same"))
    model.add(MaxPooling1D(2))

    model.add(Conv1D(512, 8, strides=1, activation='tanh', padding="same"))
    model.add(Conv1D(512, 8, strides=1, activation='tanh', padding="same"))
    model.add(MaxPooling1D(2))

    model.add(Conv1D(1024, 2, strides=1, activation='tanh', padding="same"))
    model.add(Conv1D(1024, 2, strides=1, activation='tanh', padding="same"))
    model.add(MaxPooling1D(2))

    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    return(model)

#将结果绘图
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.ylabel('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


if __name__ == "__main__":
    train_iter = xs_gen()
    val_iter = xs_gen(train=False)
    ckpt = ModelCheckpoint(
        filepath='D:\Desktop\data\\best_model.{epoch:02d}-{val_loss:.4f}.h5',
        monitor='val_loss', save_best_only=True, verbose=1
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
    #结果输出
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







