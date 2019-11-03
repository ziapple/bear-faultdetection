from scipy.io import loadmat
import numpy as np
import random
import os
from PyEMD import EEMD

# 采样频率
TIME_PERIODS = 400
# 标签类别
LABEL_SIZE = 10
# 转化成IMF分量长度
IMF_LENGTH = 7
# IMF样本保留分量长度
IMF_X_LENGTH = 3


def read_data():
    """
    读取data下面所有文件
    :param time_steps: 采样率，输入数据的长度，步长
    :return:
    """
    eemd = EEMD()
    emd = eemd.EMD
    emd.extrema_detection = "parabol"
    for item in os.listdir('../data'):
        # 每读取一个文件，保存一个
        print("start...{%s}" % item)
        # 获取文件名
        short_name, _ = os.path.splitext(item)
        time_series = read_matdata('../data/' + item)
        # 将故障样本拆分为训练样本长度
        input_length = time_series.shape[0]//TIME_PERIODS
        # 三维矩阵，记录每个输入样本长度，每个分量，信号信息
        x = np.zeros((input_length, IMF_LENGTH, TIME_PERIODS))
        # 获取序列最大长度，去掉信号后面的余数
        idx_last = -(time_series.shape[0] % TIME_PERIODS)
        # 切片分割(input_length, time_steps)
        clips = time_series[:idx_last].reshape(-1, TIME_PERIODS)
        # 对每个样本做eemd处理
        for i in range(clips.shape[0]):
            eimfs = eemd.eemd(clips[i])
            print("start emf...%s-%d" % (item, i))
            print(eimfs)
            x[i] = eimfs[0:IMF_LENGTH]
        # 将每个输入样本拉平，存储
        b = x.reshape(input_length, -1)
        np.savetxt('../emd_data/' + short_name + '.txt', b)
        print(x)


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


def read_emd_1():
    """
    读取转化后的emd_data文件,一维卷积，准确率很低
    :return:
    """
    x_train = np.zeros((0, TIME_PERIODS))
    x_test = np.zeros((0, TIME_PERIODS))
    y_train = []
    y_test = []
    i = 0
    for item in os.listdir("../emd_data/" + str(TIME_PERIODS)):
        print("read %s" % item)
        # x = IMF_LENGTH * TIME_PERIOD,每个分量作为一个样本
        x = np.loadtxt("../emd_data/" + str(TIME_PERIODS) + "/" + item)
        x = x.reshape(-1, IMF_LENGTH, TIME_PERIODS)
        # 取第一维主向量IMF，准确率能达到95%以上
        x = x[:, :1, :]
        # 每个二维样本，作为行展开，得到x.shape[0] * x.shape[1]个样本
        x = x.reshape(x.shape[0] * x.shape[1], -1)
        n = x.shape[0]
        # 按3/4比例为训练数据，1/4为测试数据
        n_split = int((3 * n / 4))
        # 二维数组填充,增量式填充
        x_train = np.vstack((x_train, x[:n_split]))
        x_test = np.vstack((x_test, x[n_split:]))
        # [0]+[1] = [0, 1],不断累积标签
        y_train += [i] * n_split
        y_test += [i] * (x.shape[0] - n_split)
        i += 1
    return x_train, y_train, x_test, y_test


def read_emd_to_normal():
    """
    读取转化后的emd_data文件,每个样本按照(IMF_LENGTH, TIME_PERIODS)样本返回
    :return:
    """
    x_train = np.zeros((0, IMF_X_LENGTH, TIME_PERIODS))
    x_test = np.zeros((0, IMF_X_LENGTH, TIME_PERIODS))
    y_train = []
    y_test = []
    i = 0
    for item in os.listdir("../emd_data/" + str(TIME_PERIODS)):
        print("read %s" % item)
        # x = IMF_LENGTH * TIME_PERIOD,每个分量作为一个样本
        x = np.loadtxt("../emd_data/" + str(TIME_PERIODS) + "/" + item)
        x = x.reshape(-1, IMF_LENGTH, TIME_PERIODS)
        # 取前IMF_X_LENGTH个
        x = x[:, :IMF_X_LENGTH, :]
        n = x.shape[0]
        # 按3/4比例为训练数据，1/4为测试数据
        n_split = int((3 * n / 4))
        # 二维数组填充,增量式填充
        x_train = np.vstack((x_train, x[:n_split]))
        x_test = np.vstack((x_test, x[n_split:]))
        # [0]+[1] = [0, 1],不断累积标签
        y_train += [i] * n_split
        y_test += [i] * (x.shape[0] - n_split)
        i += 1
    return x_train, y_train, x_test, y_test


def read_emd_to_img():
    """
    读取转化后的emd_data文件,类似图片处理，每个样本按照(1, TIME_PERIODS, IMF_LENGTH)维度返回
    :return:
    """
    x_train = np.zeros((0, 1, TIME_PERIODS, IMF_X_LENGTH))
    x_test = np.zeros((0, 1, TIME_PERIODS, IMF_X_LENGTH))
    y_train = []
    y_test = []
    i = 0
    for item in os.listdir("../emd_data/" + str(TIME_PERIODS)):
        print("read %s" % item)
        # x = IMF_LENGTH * TIME_PERIOD,每个分量作为一个样本
        x = np.loadtxt("../emd_data/" + str(TIME_PERIODS) + "/" + item)
        # 读取样本，还原
        x = x.reshape(-1, IMF_LENGTH, TIME_PERIODS)
        # 将IMF作为深度变换（类似图片的RGB），转化成1*TIME_PERIODS*IMF_LENGTH
        x_1 = np.zeros((x.shape[0], 1, TIME_PERIODS, IMF_LENGTH))
        for j in range(x.shape[0]):
            # 每个样本做一个转置
            x_1[j, 0] = x[i].reshape(IMF_LENGTH, TIME_PERIODS).T
        # 取前IMF_X_LENGTH个
        x_1 = x_1[:, :, :, :IMF_X_LENGTH]
        n = x_1 .shape[0]
        # 按3/4比例为训练数据，1/4为测试数据
        n_split = int((3 * n / 4))
        # 二维数组填充,增量式填充
        x_train = np.vstack((x_train, x_1[:n_split]))
        x_test = np.vstack((x_test, x_1[n_split:]))
        # [0]+[1] = [0, 1],不断累积标签
        y_train += [i] * n_split
        y_test += [i] * (x.shape[0] - n_split)
        i += 1
    return x_train, y_train, x_test, y_test


def test_x_reshape():
    """
    测试x存储和读取转换
    :return:
    """
    # 初始化一个随机样本
    input_length = 2
    x = np.zeros((input_length, IMF_X_LENGTH, TIME_PERIODS))
    for i in range(x.shape[0]):
        # 模拟emd
        x_i = np.random.random((IMF_X_LENGTH, TIME_PERIODS))
        x[i] = x_i

    # 将每个输入样本拉平，存储
    b = x.reshape(input_length, -1)

    # 读取样本，还原
    c = b.reshape(-1, IMF_X_LENGTH, TIME_PERIODS)
    # 比较第一个样本的IMF分量
    # print(np.equal(x[0, 0], c[0, 0]))

    # 将IMF作为深度变换，转化成1*TIME_PERIODS*IMF_LENGTH
    d = np.zeros((input_length, 1, TIME_PERIODS, IMF_X_LENGTH))
    for i in range(b.shape[0]):
        # 每个样本做一个转置，再叠加
        a = b[i].reshape(IMF_X_LENGTH, TIME_PERIODS).T
        d[i, 0] = a
    print(np.equal(x[0, 0], d[0, 0, :, 0]))


if __name__ == "__main__":
    # x_train, y_train, x_test, y_test = read_emd_2()
    # print(x_train.shape)
    read_emd_to_img()
