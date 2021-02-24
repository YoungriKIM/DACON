# https://machinelearningmastery.com/multi-label-classification-with-deep-learning/
# 딥러닝으로 다중라벨분류모델 만들기 공부

# mlp for multi-label classification
import numpy as np
from numpy import mean
from numpy import std
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import RepeatedKFold, train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score

# get the dataset
def get_dataset():
	X, y = make_multilabel_classification(n_samples=500, n_features=65536, n_classes=26, n_labels=15, random_state=1)
	return X, y

X, y = get_dataset()

# print(X.shape)
# print(y.shape)

# print(X[0])
# print(y[0])

# print(np.min(X), np.max(X))

# get the model
def get_model(n_inputs, n_outputs):
	model = Sequential()
	model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(n_outputs, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model



X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=311)

model = get_model(X.shape[1], y.shape[1])
model.fit(X_train, y_train, verbose=1, epochs=100)
# make a prediction on the test set
yhat = model.predict(X_test)
# round probabilities to class labels
yhat = yhat.round()
# calculate accuracy
acc = accuracy_score(y_test, yhat)
# store result
print('>%.3f' % acc)
print(yhat)
