from scipy.io import loadmat
import numpy as np
import random
import os
from keras.utils import to_categorical
from scipy.fftpack import fft
import pywt
from PyEMD import EEMD

# 采样频率
TIME_PERIODS = 6000
# 标签类别
LABEL_SIZE = 10


def read_data(x_len=TIME_PERIODS, label_size=LABEL_SIZE):
    """
    读取data下面所有文件
    :param x_len: 采样率，输入数据的长度
    :param label_size 标签大小
    :return:
    """
    x_train = np.zeros((0, x_len))
    x_test = np.zeros((0, x_len))
    y_train = []
    y_test = []
    i = 0
    for item in os.listdir('../data'):
        print(item)
        time_series = read_matdata('../data/' + item)
        # 获取序列最大长度,时序数据有122571个的采集点，采样率为12K，采集了10秒,id_last=-571
        idx_last = -(time_series.shape[0] % x_len)
        # 切片分割,clips.shape=(122, 1000),从后往前取
        clips = time_series[:idx_last].reshape(-1, x_len)
        n = clips.shape[0]
        # 按3/4比例为训练数据，1/4为测试数据
        n_split = int((3 * n / 4))
        # 二维数组填充,增量式填充
        x_train = np.vstack((x_train, clips[:n_split]))
        x_test = np.vstack((x_test, clips[n_split:]))
        # [0]+[1] = [0, 1],不断累积标签
        y_train += [i] * n_split
        y_test += [i] * (clips.shape[0] - n_split)
        i += 1

    x_train, y_train = _shuffle(x_train, y_train)
    x_test, y_test = _shuffle(x_test, y_test)
    # y做one-hot处理
    y_train = to_categorical(y_train, label_size)
    y_test = to_categorical(y_test, label_size)

    return x_train, y_train, x_test, y_test


def read_matdata(fpath):
    """
       读取DE_time驱动端的振动数据
       DE - drive end accelerometer data
       FE - fan end accelerometer data
       BA - base accelerometer data
       time - time series data
       RPM- rpm during testing
    """
    mat_dict = loadmat(fpath)
    # 过滤DE_time这一列
    fliter_i = filter(lambda x: 'DE_time' in x, mat_dict.keys())
    # 构造数组
    fliter_list = [item for item in fliter_i]
    # 获取第一列
    key = fliter_list[0]
    # 获取n*1二维矩阵的第1列, time_serries.shape=(122571,)
    time_series = mat_dict[key][:, 0]
    return time_series


# 给定数组重新排列
def _shuffle(x, y):
    # shuffle training samples
    index = list(range(x.shape[0]))
    random.Random(0).shuffle(index)
    x = x[index]
    y = tuple(y[i] for i in index)
    return x, y


def x_fft(x):
    """
    傅里叶变换
    :param x 样本，每个点代表t时刻幅值
    :return: 每个点代表n频率下的幅值，由于傅里叶变换后对称性，取一半
    """
    return abs(fft(x)/2)


def x_wavelet(x):
    """
    小波变换
    :param x
    :return:
    """
    # cgau8为小波函数
    cwtmatr1, freqs1 = pywt.cwt(x_train, np.arange(1, 500), 'cgau8', 1 / 500)
    return abs(cwtmatr1)


def x_eemd(x):
    """
    eemd变换
    :param x: 代表采样数据[m,n],m代表每个周期的样本，n代表采样点
    :return: 返回每个样本IMF分量，[m,IMF,n]
    """
    eemd = EEMD()
    emd = eemd.EMD
    emd.extrema_detection = "parabol"
    for i in range(x.shape[0]):
        eIMFs = eemd.eemd(x[i])
        print(eIMFs)


if __name__ == "__main__":
    x_train, _, _, _ = read_data(x_len=400)
    x_eemd(x_train)