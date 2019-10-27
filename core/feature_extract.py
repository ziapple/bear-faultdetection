# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 21:35:24 2018
@author: 13260
"""
import os
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np


def feature_extraction(filename, save_path):
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
    img_path = filename
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    # 获取VGG19全连接层特征
    fc2 = model.predict(x)
    # 保存特征文件
    np.savetxt(save_path + '.txt', fc2, fmt='%s')


def read_image(rootdir, save_path):
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    # print(list)
    # files = []
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        # print(path)
        # subFiles = []
        for file in os.listdir(path):
            # subFiles.append(file)
            savePath = os.path.join(save_path, file[:-4])
            # print(file)
            filename = os.path.join(path, file)
            feature_extraction(filename, savePath)
            print("successfully saved " + file[:-4] + ".txt !")


if __name__ == '__main__':
    # 加载VGG19模型及参数
    base_model = VGG19(weights='imagenet', include_top=True)
    print("Model has been onload !")
    # 图片路径
    rootdir = 'D:/python/bear-faultdetect/cwru/data/img'
    # 提取特征文件保存路径
    save_path = "D:/python/bear-faultdetect/cwru/data/img"
    read_image(rootdir, save_path)
    print("work has been done !")
