import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
import pandas as pd
from xlutils.copy import copy
import xlrd as xr


mpl.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号


for n in range(0,4):
    df = pd.read_excel(r'D:\Desktop\two.xls', sheet_name='test12', names=['value'], usecols=[n+1])
    data = df.values


    y = []

    for i in data:
        y.append(i[0])

    fft_y = fft(y)  # 快速傅里叶变换

    N = len(y)
    x = np.arange(N)  # 频率个数
    half_x = x[range(int(N / 2))]  # 取一半区间

    abs_y = np.abs(fft_y)  # 取复数的绝对值，即复数的模(双边频谱)
    angle_y = np.angle(fft_y)  # 取复数的角度
    normalization_y = abs_y / N  # 归一化处理（双边频谱）
    normalization_half_y = normalization_y[range(int(N / 2))]  # 由于对称性，只取一半区间（单边频谱）

    # plt.plot(x, normalization_y, 'blue')
    # plt.title('单边振幅谱(归一化)', fontsize=9, color='blue')
    # plt.show()
    file = "D:\Desktop\mytext.xls"
    oldwb = xr.open_workbook(file)  # 打开工作簿
    newwb = copy(oldwb)  # 复制出一份新工作簿
    newws = newwb.get_sheet(-1 + 12)  # 获取指定工作表，0表示实际第一张工作表
    for i in range(len(normalization_half_y)):
        newws.write(i + 1, n + 1, normalization_half_y[i])  # 把列表a中的元素逐个写入第一列，0表示实际第1列,i+1表示实际第i+2行
    newwb.save(file)  # 保存修改


