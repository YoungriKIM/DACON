# 0122-1 를 가져와서 48개 스플릿을 1스플릿으로 나누기
# 할 때 마다 저장 파일 명 바꿔라~!

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Input, Flatten, MaxPooling1D, Dropout, Reshape, SimpleRNN, LSTM, LeakyReLU, GRU, Conv2D, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.backend import mean, maximum
import os
import glob
import random
import tensorflow.keras.backend as K

#===================================================================
# train 데이터 불러옴

# 원하는 열만 가져오기
dataset = pd.read_csv('../data/csv/dacon1/train/train.csv', index_col=None, header=0)
# print(dataset.shape)
x_train = dataset.iloc[:,[1,3,4,5,6,7,8]]
print(x_train.shape)      #(52560, 7)

#===================================================================
# 81개의 all_test 데이터 48개(1일치) 불러와서 합치기
def preprocess_data(data):
    temp = data.copy()
    return temp.iloc[-48:,[1,3,4,5,6,7,8]]

df_test = []

for i in range(81):
    file_path = '../data/csv/dacon1/test/' + str(i) + '.csv'
    temp = pd.read_csv(file_path)
    temp = preprocess_data(temp)
    df_test.append(temp)

all_test = pd.concat(df_test)
print(all_test.shape)   #(3888, 7)


#===================================================================
# # GHI라는 기준 추가
def Add_features(data):
    data['cos'] = np.cos(np.pi/2 - np.abs(data['Hour'] % 12 - 6)/6 * np.pi/2)
    # pi = 원주율, abs = 절대값
    data.insert(1, 'GHI', data['DNI'] * data['cos'] + data['DHI'])
    # 데이터를 넣어줄건데 1열에(기존 열은 오른쪽으로 밀림), 'GHI'명으로, 마지막의 수식으로 나온 값을
    data.drop(['cos'], axis=1, inplace = True)
    #'cos' 열을 삭제를 할 것 이고. 이 삭제한 데이터프레임으로 기존 것을 대체하겠다.
    return data

x_train = Add_features(x_train)     # 트레인에 붙여줌
all_test = Add_features(all_test).values    # 테스트에 붙여줌

print(x_train.shape)      #(52560, 8)   #하나씩 붙은 모습
print(all_test.shape)     #(3888, 8)

#===================================================================
# train에 다음날, 다다음날의 TARGET을 오른쪽 열으로 붙임
day_7 = x_train['TARGET'].shift(-48)      #다음날
day_8 = dataset['TARGET'].shift(-48*2)    #다다음날

x_train = pd.concat([x_train, day_7, day_8], axis=1)
# dataset2.columns = ['Hour', 'GHI', 'DHI', 'DNI', 'WS', 'RH','T','TARGET','TARGET+1','TARGET+2']
x_train = x_train.iloc[:-96,:]  # 마지막 2일은 데이터가 비니까 빼준다

print(x_train.shape)       #(52464, 10)       # 8개는 기준일 +GHI / +day7 + day8

#===================================================================
# x_train을 RNN식으로 데이터 자르기
aaa = x_train.values

def split_xy(aaa, x_row, x_col, y_row, y_col):
    x, y = list(), list()
    for i in range(len(aaa)):
        if i > len(aaa)-x_row:
            break
        tmp_x = aaa[i:i+x_row, :x_col]
        tmp_y = aaa[i:i+x_row, x_col:x_col+y_col]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

# print(x, '\n\n', y)
x_train, y_train = split_xy(aaa, 1,8,1,2)     # 30분씩 RNN식으로 자름
print(x_train.shape)                    #(52464, 1, 8)
print(y_train.shape)                    #(52464, 1, 2)

#===================================================================
# 데이터 전처리 : 준비 된 데이터 x_train / y_train / all_test
# 1) 트레인테스트분리 / 2) 민맥스or스탠다드 / 3) 모델에 넣을 쉐잎

# 1) 2차원으로 만들어서 트레인테스트분리
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
y_train = y_train.reshape(y_train.shape[0], y_train.shape[1]*y_train.shape[2])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=311)
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, shuffle=True, random_state=311)

# 2) 스탠다드 스케일러
from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)
all_test = scaler.transform(all_test)

# 3) 모델에 넣을 쉐잎
# for conv2D
x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
x_val = x_val.reshape(x_val.shape[0], 1, x_val.shape[1])
x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
all_test = all_test.reshape(all_test.shape[0], 1, all_test.shape[1])

# y_train = y_train.reshape(y_train.shape[0], y_train.shape[1])
# y_val = y_val.reshape(y_val.shape[0], y_val.shape[1])
# y_test = y_test.reshape(y_test.shape[0], y_test.shape[1])

# print(x_train.shape)
# print(x_val.shape)
# print(x_test.shape)
# print(all_test.shape)
# print(y_train.shape)
# print(y_val.shape)
# print(y_test.shape)
# (33576, 1, 8)
# (8395, 1, 8)
# (10493, 1, 8)
# (3888, 1, 8)
# (33576, 2)
# (8395, 2)
# (10493, 2)

#===================================================================
#퀀타일 로스 적용된 모델 구성 + 컴파일, 훈련까지

def quantile_loss(q, y_true, y_pred):
    err = (y_true - y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)
# mean = 평균
# K 를 tensorflow의 백앤드에서 불러왔는데 텐서형식의 mean을 쓰겠다는 것이다.

qlist = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
subfile = pd.read_csv('../data/csv/dacon1/sample_submission.csv')


def mymodel():
    model = Sequential()
    model.add(Conv1D(96, 2, input_shape=(x_train.shape[1], x_train.shape[2]), padding='same', activation='relu'))
    model.add(Conv1D(96, 2, padding='same'))
    model.add(Conv1D(96, 2, padding='same'))
    model.add(Flatten())
    model.add(Dense(96))
    model.add(Dense(96))
    model.add(Dense(48))
    model.add(Dense(2))
    return model

# model.summary()

for q in qlist:
    patience = 8
    print(str(q)+'번째 훈련')
    model = mymodel()
    model.compile(loss = lambda y_true, y_pred: quantile_loss(q, y_true, y_pred), optimizer='adam', metrics=['mse'])
    stop = EarlyStopping(monitor ='val_loss', patience=patience, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=patience/2, factor=0.5)
    filepath = f'../data/modelcheckpoint/dacon_train_0124_1_{q:.1f}.hdf5'
    check = ModelCheckpoint(filepath = filepath, monitor = 'val_loss', save_best_only=True, mode='min') #앞에 f를 붙여준 이유: {}안에 변수를 넣어주겠다는 의미
    hist = model.fit(x_train, y_train, epochs=500, batch_size=48, verbose=1, validation_split=0.2, callbacks=[stop, reduce_lr])#, check])
    
    # 평가, 예측
    result = model.evaluate(x_test, y_test, batch_size=48)
    print('loss: ', result[0])
    print('mae: ', result[1])
    y_predict = model.predict(all_test)
    print(y_predict.shape)  #(3888, 2)
    
    
    # 예측값을 submission에 넣기
    y_predict = pd.DataFrame(y_predict)
    y_predict2 = pd.concat([y_predict], axis=1)
    y_predict2[y_predict2<0] = 0
    y_predict3 = y_predict2.to_numpy()
        
    print(str(q)+'번째 지정')
    subfile.loc[subfile.id.str.contains('Day7'), 'q_' + str(q)] = y_predict3[:,0].round(2)
    subfile.loc[subfile.id.str.contains('Day8'), 'q_' + str(q)] = y_predict3[:,1].round(2)

    # print(subfile.head())

subfile.to_csv('../data/csv/dacon1/sub_0124_1-4.csv', index=False)

#===================================================================
print('(ง˙∇˙)ว {오늘 안에 조지고만다!!!]')

# split 48 conv1d   loss:  0.8572525382041931
# size 48,1,8 Conv2D    loss:  0.9139912128448486
# size 1,48,8 Conv2D    loss:  0.8800864219665527
# 0121-5 : loss:  0.7130502462387085    0121-5 2.72518
# 섭미션 수정 후 
# ing  loss:  0.7192889451980591    1.92426	 기쁨의 땐스 ^^
# 3888,1,1,8 > 3888,2 > 저장명 0124_1-1  loss:  0.8100641965866089  mae:  253.9652862548828  > 1.9263783575	
# 3888,1,8,1 > 3888,2 > 저장명 0124_1-2   loss:  0.8455850481987   mae:  262.62847900390625
# 3888,8,1,1 > 3888,2 > 저장명 0124_1-3    loss:  0.901399970054626  mae:  280.4082336425781
# 3888,1,8 > 3888,2 저장명 0124_1-4   loss:  0.8110382556915283    mae:  254.71890258789062  > 1.9284723357	
