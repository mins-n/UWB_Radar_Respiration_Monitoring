# regression for MNIST Data set
import sys
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, BatchNormalization
from sklearn.model_selection import train_test_split
from keras.layers.convolutional import Conv1D, MaxPooling1D
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

window_size = 10
uwb_fs = 20
biopac_fs = 500
root_dir = "./../Data/"


def get_img_path(root_dir):
    file_list = []
    for (root, dirs, files) in os.walk(root_dir):
        for file_name in files:
            if (file_name.endswith("_gray.npy")):
                file_list.append(root + "/" + file_name)
    return file_list


def get_peak_path(root_dir):
    file_list = []
    for (root, dirs, files) in os.walk(root_dir):
        for file_name in files:
            if (file_name.endswith("_ref.npy")):
                file_list.append(root + "/" + file_name)
    return file_list


np.random.seed(7)

print('Python version : ', sys.version)
print('TensorFlow version : ', tf.__version__)
print('Keras version : ', keras.__version__)

data_path_list = get_img_path(root_dir)
ref_path_list = get_peak_path(root_dir)
#np.save("1", data_path_list)
#np.save("2",ref_path_list)
if len(data_path_list) != len(ref_path_list):
    print("데이터 크기 불일치")
    sys.exit("오류")

data_list = []
ref_list = []

for data_path in data_path_list:
    data_list.append(np.load(data_path))
for ref_path in ref_path_list:
    ref_list.append(np.load(ref_path))
data_list = np.reshape(np.array(data_list),(len(data_list)*110,31,200))
ref_list = np.reshape(np.array(ref_list),(1,len(ref_list)*110))
ref_list = ref_list[0]


x_train, x_test, y_train, y_test = train_test_split(data_list, ref_list, test_size=0.1, shuffle=False)

print('x_train : ', x_train[0])
print('y_train : ', y_train[0])
img_rows = 31
img_cols = 200

input_shape = (img_rows, img_cols, 1)
batch_size = 110
epochs = len(data_list)

model = Sequential()

model.add(BatchNormalization())

model.add(Conv1D(64, kernel_size=9, strides=2, padding='same',
                 activation='relu',
                 input_shape=input_shape))

model.add(MaxPooling1D(pool_size=2, strides=2))
model.add(Conv1D(128, kernel_size= 5, strides=2, activation='relu', padding='same'))
model.add(Conv1D(256, kernel_size=3, strides=2, activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=2, strides=2))
model.add(LSTM(512, recurrent_dropout=0.1))
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