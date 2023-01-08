
# regression for MNIST Data set
import sys
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, BatchNormalization
from keras.layers.convolutional import Conv1D, MaxPooling1D

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7)

print('Python version : ', sys.version)
print('TensorFlow version : ', tf.__version__)
print('Keras version : ', keras.__version__)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print('x_train : ', x_train[0])
print('y_train : ', y_train[0])
img_rows = 31
img_cols = 400

input_shape = (img_rows, img_cols, 1)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

batch_size = 128
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
