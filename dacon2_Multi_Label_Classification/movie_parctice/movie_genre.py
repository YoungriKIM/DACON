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

# x_train 불러오기 =====================================
# x = np.load('../data/npy/dacon12/dirty_mnist_train_all(50000).npy').astype('float32')
# print(x.shape)  #(50000, 256, 256, 1)

# dataset_data = x.reshape(x.shape[0], x.shape[1]*x.shape[2])

# y_train 불러오기 =====================================
y_train = pd.read_csv('../dacon12/data/dirty_mnist_2nd_answer.csv', index_col=0)
print(y_train.head())