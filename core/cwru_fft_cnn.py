import keras
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from cwru.core import cwru_input


# 迭代次数
EPOCHS = 50
# 每批次读取的训练数据集大小
BATCH_SIZE = 20


# 建立Sequential网络模型
def build_model(input_shape=(cwru_input.TIME_PERIODS,), num_classes=cwru_input.LABEL_SIZE):
    model_inference = Sequential()
    # 输入层变成(6000, 1)二维矩阵
    model_inference.add(Reshape((cwru_input.TIME_PERIODS, 1), input_shape=input_shape))
    model_inference.add(Conv1D(16, 8, strides=2, activation='relu', input_shape=(cwru_input.TIME_PERIODS, 1)))
    model_inference.add(Conv1D(16, 8, strides=2, activation='relu', padding="same"))
    model_inference.add(MaxPooling1D(2))
    model_inference.add(Conv1D(32, 4, strides=2, activation='relu', padding="same"))
    model_inference.add(Conv1D(32, 4, strides=2, activation='relu', padding="same"))
    model_inference.add(MaxPooling1D(2))
    model_inference.add(Conv1D(256, 4, strides=2, activation='relu', padding="same"))
    model_inference.add(Conv1D(256, 4, strides=2, activation='relu', padding="same"))
    model_inference.add(MaxPooling1D(2))
    model_inference.add(Conv1D(512, 2, strides=1, activation='relu', padding="same"))
    model_inference.add(Conv1D(512, 2, strides=1, activation='relu', padding="same"))
    model_inference.add(MaxPooling1D(2))
    model_inference.add(GlobalAveragePooling1D())
    model_inference.add(Dropout(0.3))
    """
    model_inference.add(Flatten())
    model_inference.add(Dropout(0.3))
    """
    model_inference.add(Dense(256, activation='relu'))
    # Dense标准的一维全连接层
    model_inference.add(Dense(num_classes, activation='softmax'))
    return model_inference


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = cwru_input.read_data()
    # 傅里叶变换
    x_train, x_test = cwru_input.x_fft(x_train, x_test)
    print(x_train.shape)
    ckpt = keras.callbacks.ModelCheckpoint(
        filepath='../model/best_model.{epoch:02d}-{val_loss:.4f}.h5',
        monitor='val_loss', save_best_only=True,  verbose=1)

    model = build_model()
    opt = Adam(0.0002)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    print(model.summary())

    model.fit(
        x=x_train,
        y=y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.3,
        callbacks=[ckpt],
        )
    model.save("finishModel.h5")
