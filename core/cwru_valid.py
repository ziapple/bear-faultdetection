from keras.models import *
from cwru.core import cwru_input
import numpy as np


if __name__ == "__main__":
    # 傅里叶转换能够有效提升神经网络的准确率
    _, _, x_test, y_test = cwru_input.read_data(x_fft=True)
    model = load_model('finishModel.h5')
    _y = model.predict(x_test)
    acc = np.equal(np.argmax(_y, axis=1), np.argmax(y_test, axis=1))
    print(np.sum(acc)/y_test.shape[0])
