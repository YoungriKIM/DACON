# keras_multilabel_example_use 가져와서 conv2d로 쓰기



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from numpy import mean
from numpy import std
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import RepeatedKFold
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Conv2D, Flatten, BatchNormalization, Dropout
from sklearn.metrics import accuracy_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# npy 불러와서 x로 지정 + scaling =====================================
x = np.load('../data/npy/dacon12/dirty_mnist_train_all(50000).npy').astype('float32')
# print(x.shape)  #(50000, 256, 256, 1)

# dataset_data = x.reshape(x.shape[0], x.shape[1]*x.shape[2])

# y 불러와서 npy로 변환 =====================================
y = pd.read_csv('../dacon12/data/dirty_mnist_2nd_answer.csv', index_col=0).values

# print(dataset_data.shape)
# print(dataset_target.shape)
# (500, 65536)
# (500, 26)


# ----------------------------------------
# 함수 먼저 정의
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=311)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
# (40000, 256, 256, 1)
# (10000, 256, 256, 1)
# (40000, 26)
# (10000, 26)


# get the model

n_inputs = x_train.shape[1:]
n_outputs = y_train.shape[1]

model = Sequential()
model.add(Conv2D(32, kernel_size=(128,128), input_shape=n_inputs, strides=16, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(64,64), input_shape=n_inputs, strides=8, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(n_outputs, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])

stop = EarlyStopping(monitor='val_loss', patience=16, mode='auto')
file_path = '../data/h5/daon12/multi_01-{val_loss:.4f}.h5'
mc = ModelCheckpoint(monitor='val_loss', mode = 'auto', save_best_only=True, filepath=file_path)

model.fit(x_train, y_train, batch_size=32, verbose=1, epochs=5, validation_split=0.2, callbacks=[stop, mc])

loss = model.evaluate(x_test, y_test)
print('loss, acc: ', loss)
print(model.predict(x_test).shape)
print(model.predict(x_test))