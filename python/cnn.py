# regression for MNIST Data set
import sys
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, BatchNormalization
from sklearn.model_selection import train_test_split
from keras.layers.convolutional import Conv1D, MaxPooling1D
from PIL import image
import numpy as np
import matplotlib.pyplot as plt
import os

window_size = 10
uwb_fs = 20
biopac_fs = 500
root_dir = "./../Data/"


def get_img_path(root_dir):
    file_list = []
    for (root, dirs, files) in os.walk(root_dir):
        for file_name in files:
            if (file_name.endswith("_gray.jpg")):
                file_list.append(root + "/" + file_name)
    return file_list


def get_peak_path(root_dir):
    file_list = []
    for (root, dirs, files) in os.walk(root_dir):
        for file_name in files:
            if (file_name.endswith("_data.npy")):
                file_list.append(root + "/" + file_name)
    return file_list

def generate_dataset():
    img = np.zeros((31, 200))
    img.reshape(1, 31, 200)

    img_path_list = get_img_path(root_dir)
    for img_path in img_path_list:
        img.append([])
        load_img = image.open(img_path)
        img_data = np.array(load_img)
        for j in range(110):
            tmp = img_data[:][j * uwb_fs: (10 + j) * uwb_fs]
            tmp.reshape(1, len(tmp), len(tmp[0]))
            img = tmp.concatenate((img, tmp), axis=0)
    img = np.delete(img, 0, axis=0)
    return img


def generate_ref():
    ref_list = []
    ref_path_list = get_peak_path(root_dir)
    for ref_path in ref_path_list:
        load_ref = np.load(ref_path)
        ref_list.append([])
        for j in range(110):
            tmp = load_ref
            tmp = tmp[tmp > j * biopac_fs]
            tmp = tmp[tmp < (10 + j) * biopac_fs]
            ref_list[i].append(len(tmp))
    return ref_list


np.random.seed(7)

print('Python version : ', sys.version)
print('TensorFlow version : ', tf.__version__)
print('Keras version : ', keras.__version__)

data = generate_dataset()
answer = generate_ref()

x_train, x_test, y_train, y_test = train_test_split(data, answer, test_size=0.1, shuffle=False)

print('x_train : ', x_train[0])
print('y_train : ', y_train[0])
img_rows = 31
img_cols = 200

input_shape = (img_rows, img_cols, 1)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

batch_size = 110
epochs = 100

y_train = y_train.reshape(len(x_train), 1)
y_test = y_test.reshape(len(x_test), 1)

model = Sequential()
model.add(BatchNormalization())
model.add(Conv1D(64, kernel_size=(1, 9), strides=(1, 2), padding='same',
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling1D(pool_size=(2, 1), strides=(1, 2)))

model.add(Conv1D(128, kernel_size=(1, 5), strides=(1, 2), activation='relu', padding='same'))
model.add(Conv1D(256, kernel_size=(1, 3), strides=(1, 2), activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=(2, 1)))

model.add(LSTM(512, activation='sigmoid'))
model.add(Flatten(64, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', learning_rate=0.001, metrics=["accuracy"])

hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size)

plt.figure(figsize=(12, 8))
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.legend(['loss', 'val_loss', 'accuracy', 'val_accuracy'])
plt.show()

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

n = 0
plt.imshow(x_test[n].reshape(28, 28), cmap='Greys', interpolation='nearest')
plt.show()
print(model.predict(x_test[n].reshape(1, 28, 28, 1)))