import numpy as np
import matplotlib.pyplot as plt
from cwru.core import cwru_input
from scipy.fftpack import fft
import pywt

# 采用频率6000
length = cwru_input.TIME_PERIODS


def draw():
    t = np.linspace(0, 1, length, endpoint=False)
    time_serials = cwru_input.read_matdata('../data/12k_Drive_End_B007_0_118.mat')
    print(time_serials)
    y = time_serials[0: length]
    # 图1画出原始频谱
    plt.figure(figsize=(9, 15))
    plt.subplot(311)
    plt.plot(t, y)
    plt.title('signal_1 in time domain')
    plt.xlabel('Time/second')

    # 图2画出fft变换后的频谱
    # 采样定理告诉我们，采样频率要大于信号频率的两倍，FFT之后的频谱分辨率能够到0.5HZ
    # FFT变化的物理意义http://blog.sina.com.cn/s/blog_640029b301010xkv.html
    """
    假设采样频率为Fs，信号频率F，采样点数为N。那么FFT之后结果就是一个为N点的复数。每一个点就对应着一个频率
    点。这个点的模值，就是该频率值下的幅度特性
    """
    y1 = abs(fft(y))
    plt.subplot(312)
    plt.plot(range(length), y1)
    plt.title('signal_1 in frequency domain')
    plt.xlabel('Frequency/Hz')

    # 图3画出小波变换后的频谱
    plt.subplot(313)
    # cgau8为小波函数
    cwtmatr1, freqs1 = pywt.cwt(y1, np.arange(1, 500), 'cgau8', 1 / 500)
    plt.contour(t, freqs1, abs(cwtmatr1))
    plt.title('signal_1 in time domain')
    plt.xlabel('Time/second')
    plt.ylabel('Frequency/Hz')

    plt.show()


if __name__ == "__main__":
    draw()