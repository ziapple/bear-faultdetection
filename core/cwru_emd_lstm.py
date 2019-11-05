import keras
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.utils import to_categorical
from core import cwru_input_emd

# 迭代次数
EPOCHS = 50
# 每批次读取的训练数据集大小
BATCH_SIZE = 100


# 建立Sequential网络模型
def build_model(input_shape=(cwru_input_emd.IMF_X_LENGTH, cwru_input_emd.TIME_PERIODS), num_classes=cwru_input_emd.LABEL_SIZE):
    """
    输入:在Keras中，LSTM的输入shape=(samples, time_steps, input_dim)，
    其中samples表示样本数量，time_steps表示时间步长，input_dim表示每一个时间步上的维度。
    输出:
    一个是output_dim表示输出的维度，这个参数其实就是确定了四个小黄矩形中权重矩阵的大小。
    另一个可选参数return_sequence，这个参数表示LSTM返回的时一个时间序列还是最后一个
    model.add(LSTM(input_dim=1, output_dim=6,input_length=10, return_sequences=True))
    model.add(LSTM(output_dim=32,
                   input_shape=(2, 3),
                   activation='relu',
                   return_sequences=True))
    :param input_shape:
    :param num_classes:
    :return:
    """
    model = Sequential()
    model.add(LSTM(output_dim=256, input_shape=input_shape, activation='relu', return_sequences=True))
    model.add(LSTM(output_dim=256, return_sequences=True))
    model.add(LSTM(output_dim=256))
    model.add(Dense(256, activation='relu'))  # FC2 1024
    model.add(Dropout(rate=0.25))
    model.add(Dense(num_classes, activation='softmax'))  # Output 10
    print(model.summary())
    return model


def model_train():
    x_train, y_train, x_test, y_test = cwru_input_emd.read_emd_to_normal()
    print(x_train.shape)
    y_train = to_categorical(y_train, cwru_input_emd.LABEL_SIZE)
    y_test = to_categorical(y_test, cwru_input_emd.LABEL_SIZE)
    ckpt = keras.callbacks.ModelCheckpoint(filepath='../model/emd_model.{epoch:02d}-{val_loss:.4f}.h5',
                                           monitor='val_loss', save_best_only=True, verbose=1)
    model = build_model()
    opt = Adam(0.0002)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    print(model.summary())

    model.fit(x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.3, callbacks=[ckpt])
    model.save("finishModel_lstm.h5")


def model_valid():
    _, _, x_test, y_test = cwru_input_emd.read_emd_to_normal()
    y_test = to_categorical(y_test, cwru_input_emd.LABEL_SIZE)
    model = load_model('finishModel_lstm.h5')
    _y = model.predict(x_test)
    acc = np.equal(np.argmax(_y, axis=1), np.argmax(y_test, axis=1))
    print(np.sum(acc) / y_test.shape[0])


if __name__ == "__main__":
    # model_train()
    # model_valid()
    build_model()


