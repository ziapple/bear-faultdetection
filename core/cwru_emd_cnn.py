import keras
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.utils import to_categorical
from cwru.core import cwru_input_emd

# 迭代次数
EPOCHS = 50
# 每批次读取的训练数据集大小
BATCH_SIZE = 100


# 建立Sequential网络模型
def build_model(input_shape=(cwru_input_emd.TIME_PERIODS,), num_classes=cwru_input_emd.LABEL_SIZE):
    model = Sequential()
    model.add(Reshape((input_shape[0], 1), input_shape=input_shape))
    model.add(Conv1D(filters=32, kernel_size=8, strides=2, input_shape=input_shape, activation='relu', padding='same'))
    model.add(Dropout(rate=0.25))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=4, activation='relu', padding='same'))
    model.add(Dropout(0.25))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=2, activation='relu', padding='same'))
    model.add(Dropout(0.25))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())  # FC1,64个8*8转化为1维向量
    model.add(Dropout(rate=0.25))
    model.add(Dense(256, activation='relu'))  # FC2 1024
    model.add(Dropout(rate=0.25))
    model.add(Dense(num_classes, activation='softmax'))  # Output 10
    print(model.summary())
    return model


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = cwru_input_emd.read_emd()
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
    model.save("finishModel.h5")

