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
def build_model_1d(input_shape=(cwru_input_emd.TIME_PERIODS,), num_classes=cwru_input_emd.LABEL_SIZE):
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


# 建立Sequential网络模型
def build_model_2d(input_shape=(6, cwru_input_emd.TIME_PERIODS), num_classes=cwru_input_emd.LABEL_SIZE):
    model = Sequential()
    model.add(Reshape((input_shape[0], input_shape[1], 1), input_shape=input_shape))
    model.add(Conv2D(filters=8, kernel_size=(2, 64), input_shape=input_shape, activation='relu', padding='same'))
    model.add(Dropout(rate=0.25))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Conv2D(filters=16, kernel_size=(2, 32), activation='relu', padding='same'))
    model.add(Dropout(0.25))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Conv2D(filters=32, kernel_size=(2, 16), activation='relu', padding='same'))
    model.add(Dropout(0.25))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Flatten())
    model.add(Dropout(rate=0.25))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.25))
    model.add(Dense(num_classes, activation='softmax'))  # Output 10
    print(model.summary())
    return model


# 建立Sequential网络模型
def build_model_2d_img(input_shape=(1, cwru_input_emd.TIME_PERIODS, cwru_input_emd.IMF_X_LENGTH),
                       num_classes=cwru_input_emd.LABEL_SIZE):
    model = Sequential()
    model.add(Conv2D(filters=8, kernel_size=(1, 64), input_shape=input_shape, activation='relu', padding='same'))
    model.add(Dropout(rate=0.25))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Conv2D(filters=16, kernel_size=(1, 32), activation='relu', padding='same'))
    model.add(Dropout(0.25))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Conv2D(filters=32, kernel_size=(1, 16), activation='relu', padding='same'))
    model.add(Dropout(0.25))
    model.add(MaxPooling2D(pool_size=(1, 2)))
    model.add(Flatten())
    model.add(Dropout(rate=0.25))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.25))
    model.add(Dense(num_classes, activation='softmax'))  # Output 10
    print(model.summary())
    return model


def model_train():
    x_train, y_train, x_test, y_test = cwru_input_emd.read_emd_to_img()
    print(x_train.shape)
    y_train = to_categorical(y_train, cwru_input_emd.LABEL_SIZE)
    y_test = to_categorical(y_test, cwru_input_emd.LABEL_SIZE)
    ckpt = keras.callbacks.ModelCheckpoint(filepath='../model/emd_model.{epoch:02d}-{val_loss:.4f}.h5',
                                           monitor='val_loss', save_best_only=True, verbose=1)
    model = build_model_2d_img()
    opt = Adam(0.0002)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    print(model.summary())

    model.fit(x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.3, callbacks=[ckpt])
    model.save("finishModel_cnn.h5")


def model_valid():
    _, _, x_test, y_test = cwru_input_emd.read_emd_to_img()
    y_test = to_categorical(y_test, cwru_input_emd.LABEL_SIZE)
    model = load_model('finishModel_cnn.h5')
    _y = model.predict(x_test)
    acc = np.equal(np.argmax(_y, axis=1), np.argmax(y_test, axis=1))
    print(np.sum(acc) / y_test.shape[0])


if __name__ == "__main__":
    # model_train()
    model_valid()


