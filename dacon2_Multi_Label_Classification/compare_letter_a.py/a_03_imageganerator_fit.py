# 이전 엠니스트에서 a 만 훈련시킨 뒤
# test 이미지에서 a 잘라서 예측
# 한숩빠 https://github.com/bughunt88/Study/blob/main/vision2/dacon_sigmoid.py 참고

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold, train_test_split
from keras import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
import string
from tensorflow.keras.models import load_model
import cv2

### a 만 쓸거임!!!!! ###

# npy로 불러오자!  ------------------------------------------
x_train = np.load('../dacon12/data/save/m_a_train_x.npy', allow_pickle=True)
y_train = np.load('../dacon12/data/save/m_a_train_y.npy', allow_pickle=True)
x_val = np.load('../dacon12/data/save/m_a_val_x.npy', allow_pickle=True)
y_val = np.load('../dacon12/data/save/m_a_val_y.npy', allow_pickle=True)
x_test = np.load('../dacon12/data/save/m_a_test_x.npy', allow_pickle=True)
y_test = np.load('../dacon12/data/save/m_a_test_y.npy', allow_pickle=True)
print('===== load complete =====')


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(x_val.shape)
print(y_val.shape)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)


# 훈련을 시켜보자! 모델구성 --------------------
model = Sequential()
model.add(Conv2D(128, (7,7), input_shape=(x_train.shape[1:]), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (2,2), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (2,2), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(32))
model.add(Dense(2, activation='sigmoid'))
model.summary()


# 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.01), metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
stop = EarlyStopping(monitor='val_acc', patience=20, mode='auto')
lr = ReduceLROnPlateau(monitor='val_acc', factor=0.3, patience=10, mode='max')
filepath = ('../dacon12/data/save/m_a_{val_acc:.4f}.hdf5')
mc = ModelCheckpoint(filepath=filepath, save_best_only=True, verbose=1)

model.fit(x_train, y_train, epochs=1000, batch_size=32, verbose=1, validation_data=(x_val, y_val), callbacks=[stop,lr])#,mc])

#  ----------------------------------------------------------------------------------------------
loss, acc = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("acc : ", acc)

#  ----------------------------------------------------------------------------------------------
y_predict = model.predict(x_test)

#RMSE와 R2 구하기
from sklearn.metrics import mean_squared_error
def RMSE(y_test, y_predict):
      return np.sqrt(mean_squared_error(y_test, y_predict))
print('RMSE: ', RMSE(y_test, y_predict))

from sklearn.metrics import r2_score
R2 = r2_score(y_test, y_predict)
print('R2: ', R2)

#  ---------------------------------------------------------------------------------------------
y_predict = model.predict(x_test[:30])
print('y_predict_argmax: ', y_predict.argmax(axis=1)) 
print('y_test[-5:-1]_argmax: ', y_test[0:30].argmax(axis=1)) 

# loss :  0.5739638805389404
# acc :  0.9536585211753845
# RMSE:  0.20744798
# R2:  -0.514683292498817
# y_predict_argmax:  [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
# y_test[-5:-1]_argmax:  [0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]

# a가 아닌 것들이 너무 많아서 결과치에 이상 있음 데이터량 맞춰서 다시 할 것