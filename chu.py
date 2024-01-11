import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
import pandas as pd
from xlutils.copy import copy
import xlrd as xr
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号
n=1
df = pd.read_excel(r'D:\Desktop\mytext.xls', sheet_name='Sheet12', usecols=[n, n+12])
data = df.values

a = []
b = []
value = []
for i in data:
    value.append(i[0]/i[1])

plt.plot( value, 'blue')
plt.title('单边振幅谱(归一化)', fontsize=9, color='blue')
plt.show()