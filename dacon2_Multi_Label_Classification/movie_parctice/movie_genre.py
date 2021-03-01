# https://www.analyticsvidhya.com/blog/2019/04/build-first-multi-label-image-classification-model-python/
# 다중라벨분류 모델 예제 진행

# 라이브러리 불러오기
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# y_train 불러오기 =====================================
y = pd.read_csv('D:/aidata/dacon12/dirty_mnist_2nd_answer.csv', index_col=0)
print(y.shape)  #(50000, 256, 256, 1)

# x_train 불러오기 =====================================
x = np.load('../data/npy/dirty_mnist_train_all(50000).npy').astype('float32')
print(x.shape)  #(50000, 256, 256, 1)

# 전처리
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=311)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
# (45000, 256, 256, 1)
# (5000, 256, 256, 1)
# (45000, 26)
# (5000, 26)

# 모델 구성
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(x_train.shape[1:])))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(26, activation='sigmoid'))
model.summary()

# 컴파일, 훈련
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), batch_size=64)

# 평가
loss = model.evaluate(x_test, y_test)
print('loss, acc: ', loss)

# 예측
x_pred = np.load('../data/npy/dirty_mnist_test_all(5000).npy').astype('float32')
y_pred = model.predict(x_pred)

from sklearn.metrics import r2_score
R2 = r2_score(y_test, y_pred)
print('R2: ', R2)