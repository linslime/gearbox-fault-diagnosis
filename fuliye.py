import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
import pandas as pd
n=0
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号
df = pd.read_excel(r'D:\Desktop\one.xls', sheet_name='gearbox00', names=[ 'value'], usecols=[n+1])
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

plt.plot(half_x, normalization_half_y, 'blue')
plt.title('单边振幅谱(归一化)', fontsize=9, color='blue')
plt.show()
print(normalization_half_y)

