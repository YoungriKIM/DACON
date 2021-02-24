# https://machinelearningmastery.com/multi-label-classification-with-deep-learning/
# 딥러닝으로 다중라벨분류모델 만들기 공부

# keras_multilabel_exampled 이용해서 만들기!


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

dataset_data = x.reshape(x.shape[0], x.shape[1]*x.shape[2])


# y 불러와서 npy로 변환 =====================================
y = pd.read_csv('../dacon12/data/dirty_mnist_2nd_answer.csv', index_col=0).values
dataset_target = y

# print(dataset_data.shape)
# print(dataset_target.shape)
# (500, 65536)
# (500, 26)

X = dataset_data

# ----------------------------------------
# 함수 먼저 정의
# mlp for multi-label classification

# get the model
def get_model(n_inputs, n_outputs):
	model = Sequential()
	model.add(Dense(128, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(64, activation='relu'))
	model.add(Dense(32, activation='relu'))
	model.add(Dense(n_outputs, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])
	return model

# Early Stopping
stop = EarlyStopping(monitor='val_loss', patience=16, mode='auto')
file_path = '../data/h5/daon12/multi_01-{val_loss:.4f}.h5'
mc = ModelCheckpoint(monitor='val_loss', mode = 'auto', save_best_only=True, filepath=file_path)

# evaluate a model using repeated k-fold cross-validation
def evaluate_model(X, y):
	results = list()
	n_inputs, n_outputs = X.shape[1], y.shape[1]
	# define evaluation procedure
	cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
	# enumerate folds
	for train_ix, test_ix in cv.split(X):
		# prepare data
		X_train, X_test = X[train_ix], X[test_ix]
		y_train, y_test = y[train_ix], y[test_ix]
		# define model
		model = get_model(n_inputs, n_outputs)
		# fit model
		model.fit(X_train, y_train, batch_size=8, verbose=1, epochs=100, validation_split=0.2, callbacks=[stop, mc])
		# make a prediction on the test set
		yhat = model.predict(X_test)
		# round probabilities to class labels
		yhat = yhat.round()
		# calculate accuracy
		acc = accuracy_score(y_test, yhat)
		# store result
		print('>%.3f' % acc)
		results.append(acc)
	return results

# evaluate model
results = evaluate_model(X, y)
# summarize performance
print('Accuracy: %.3f (%.3f)' % (mean(results), std(results)))

# ========================================================
