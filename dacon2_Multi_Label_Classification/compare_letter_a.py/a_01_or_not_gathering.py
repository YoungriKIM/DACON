# 이전 엠니스트에서 a 만 훈련시킨 뒤
# before mnist 의 train, test csv 에서 a 만 빼고 다른 것들 저장!
# 한숩빠 https://github.com/bughunt88/Study/blob/main/vision2/dacon_sigmoid.py 참고

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims
from sklearn.model_selection import StratifiedKFold
from keras import Sequential
from keras.layers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
import string
from tensorflow.keras.models import load_model
import cv2

# train + test 둘다 하기

### a 만 쓸거임!!!!! ###

# 이전 mnist 파일에서 a(0), 나머지(1) 라벨링 주기 ------------------------------------------
before1 = pd.read_csv("../dacon12/data/before_mnist_data/train.csv")
# 필요 없는 열 버리기
before1 = before1.drop(['id','digit'],1)
# a는 1로, a아닌 건 0으로
before1['letter'] = np.where(before1['letter'] != 'A', 0, before1['letter'])
before1['letter'] = np.where(before1['letter'] == 'A', 1, before1['letter'])
print(before1.head())

# x, y 지정
x_1 = before1.iloc[:,1: ]
y_1 = before1['letter']
y = y_1.values

print(x_1.shape)
print(y_1.shape)
# (2048, 784)
# (2048,)


# # 리쉐잎 해주고
len_a = len(x_1)
pre_x = x_1.values.reshape(len_a, 28, 28, 1)

print(pre_x.shape)

# 잘 나오나 함 확인
# plt.imshow(pre_x[3])
# plt.show()
# 확인


# 이미지 전처리 ------------------------------------------
#254보다 작고 0이아니면 0으로 만들어주기
x_df2 = np.where((pre_x <= 100) & (pre_x != 0), 0, pre_x)
x_df3 = np.where((x_df2 > 0), 255, x_df2)
x = x_df3.astype('uint8')

# 이미지 팽창 > 안 해도 될듯
# x_df4 = cv2.dilate(x_df3, kernel=np.ones((2, 2), np.uint8), iterations=1)
# 블러 적용, 노이즈 제거
# x_df4 = cv2.medianBlur(src=x_df3, ksize= 5)

# 잘 나오나 함 확인
# plt.imshow(strong_x[3])
# plt.show()
# 확인

# -----------------------------------------------------
# xy 쉐이프 확인
print(x.shape)
print(y.shape)
# (2048, 28, 28, 1)
# (2048,)


# 저장 --------------------------------------------------
np.save('../dacon12/data/save/a_or_not_x.npy', arr=x)
np.save('../dacon12/data/save/a_or_not_y.npy', arr=y)
print('done')

# 제너레이터부터 다음
