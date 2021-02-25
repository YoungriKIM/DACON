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

# 이전 mnist 파일에서 a=1, 나머지=0 준 npy 불러오기 ------------------------------------------

x = np.load('../dacon12/data/save/a_or_not_x.npy', allow_pickle=True)
y = np.load('../dacon12/data/save/a_or_not_y.npy', allow_pickle=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=311)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=True, random_state=311)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
print(x_val.shape)
print(y_val.shape)
# (1310, 28, 28, 1)
# (1310,)
# (410, 28, 28, 1)
# (410,)
# (328, 28, 28, 1)
# (328,)

# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_val = to_categorical(y_val)
# y_test = to_categorical(y_test)


# -------------------------------------------------------------
# x , y 이미지제너레이터는 여기부터 -------------------------------------------------
# 데이콘 이미지: 크기, 각도, 두께 무작위 변경 됨

# train 용 이미지 제너레이터 선언
train_datagen = ImageDataGenerator(
    rescale = 1./255       # 스케일링, 흑백이니 /255
    , rotation_range=5
    , zoom_range=1.2
    ,shear_range=0.7        # 왜곡
    ,fill_mode='nearest'
)

# val 용 이미지 제너레이터 선언
val_datagen = ImageDataGenerator(rescale=1./255)

# 이미지 제너레이터 적용
xy_train = train_datagen.flow(x_train, y_train, batch_size=1500)
xy_val = val_datagen.flow(x_val, y_val, batch_size=500)
xy_test = val_datagen.flow(x_test, y_test, batch_size=500)


# 제널레이터 npy 저장
# npy로 저장하자 -----------------------------------------------------------------------------------------------------

np.save('../dacon12/data/save/m_a_train_x.npy', arr = xy_train[0][0])
np.save('../dacon12/data/save/m_a_train_y.npy', arr = xy_train[0][1])
np.save('../dacon12/data/save/m_a_val_x.npy', arr = xy_val[0][0])
np.save('../dacon12/data/save/m_a_val_y.npy', arr = xy_val[0][1])
np.save('../dacon12/data/save/m_a_test_x.npy', arr = xy_test[0][0])
np.save('../dacon12/data/save/m_a_test_y.npy', arr = xy_test[0][1])
print('===== save complete =====')



'''
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
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.01), metrics=['acc'])

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
stop = EarlyStopping(monitor='val_loss', patience=20, mode='min')
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=10, mode='min')

history = model.fit_generator(xy_train, steps_per_epoch=130, epochs=20\
                   , validation_data=xy_val, validation_steps=20, callbacks =[stop, lr], verbose=1)


evaluate = model.evaluate(x_test, y_test)
print(evaluate)



# # predict
# pred_1 = pd.read_csv("../dacon12/data/before_mnist_data/test.csv")
# pred_1 = before4.drop(['id', 'letter'],1).values
# pred_1 = pred_1[10:30]
# pred_1 = pred_1.reshape(len(pred_1), 28,28,1)
# print(model.predict(pred_1))


'''
