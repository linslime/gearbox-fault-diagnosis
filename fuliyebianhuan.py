import numpy as np
from scipy.fftpack import fft
from matplotlib.pylab import mpl
import pandas as pd
from xlutils.copy import copy
import xlrd as xr


mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

for n in range(0,4):
    df = pd.read_excel(r'D:\Desktop\one.xls', sheet_name='gearbox40', names=[ 'value'], usecols=[n+1])
    data = df.values
    y = []

    for i in data:
        y.append(i[0])

    fft_y = fft(y)  # 快速傅里叶变换

    N = len(y)
    x = np.arange(N)  # 频率个数
    half_x = x[range(int(N / 2))]  # 取一半区间

    abs_y = np.abs(fft_y)  # 双边频谱
    angle_y = np.angle(fft_y)  # 取复数的角度
    normalization_y = abs_y / N  # 归一化处理
    normalization_half_y = normalization_y[range(int(N / 2))]  # 单边频谱


    file = "D:\Desktop\mydata.xls"
    oldwb = xr.open_workbook(file)
    newwb = copy(oldwb)
    newws = newwb.get_sheet(n)
    for i in range(len(normalization_half_y)):
        newws.write(i + 1, 5, normalization_half_y[i])
    newwb.save(file)


