import pandas as pd
from keras.layers import *
from scipy.fftpack import fft, ifft
path = r'D:\Desktop\one.xls'

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



        fft_y = fft(y)  # 快速傅里叶变换

        N = len(y)
        x = np.arange(N)  # 频率个数
        half_x = x[range(int(N / 2))]  # 取一半区间

        abs_y = np.abs(fft_y)  # 取复数的绝对值，即复数的模(双边频谱)
        angle_y = np.angle(fft_y)  # 取复数的角度
        normalization_y = abs_y / N  # 归一化处理（双边频谱）
        normalization_half_y = normalization_y[range(int(N / 2))]  # 由于对称性，只取一半区间（单边频谱）
        temp = []
        for j in normalization_half_y:
            temp.append(j)

        print(len(temp))
