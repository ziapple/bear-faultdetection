import numpy as np
import matplotlib.pyplot as plt
from cwru.core import cwru_input
from scipy.fftpack import fft
import pywt

# 采用频率6000
length = cwru_input.TIME_PERIODS


# 小波变换
def draw():
    t = np.linspace(0, 1, length, endpoint=False)
    time_serials = cwru_input.read_matdata('../data/12k_Drive_End_B007_0_118.mat')
    print(time_serials.shape)
    y = time_serials[0: length]
    # 图1画出原始频谱
    plt.figure(figsize=(9, 15))
    plt.subplot(311)
    plt.plot(t, y)
    plt.title('signal_1 in time domain')
    plt.xlabel('Time/second')

    # 图3画出小波变换后的频谱
    plt.subplot(313)
    # cgau8为小波函数
    cwtmatr1, freqs1 = pywt.cwt(y, np.arange(1, 500), 'cgau8', 1 / 500)
    w = abs(cwtmatr1)
    print(w.shape)
    plt.contour(t, freqs1, abs(cwtmatr1))
    plt.title('signal_1 in time domain')
    plt.xlabel('Time/second')
    plt.ylabel('Frequency/Hz')

    plt.show()


if __name__ == "__main__":
    draw()