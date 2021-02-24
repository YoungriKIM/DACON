# https://machinelearningmastery.com/multi-label-classification-with-deep-learning/
# 딥러닝으로 다중라벨분류모델 만들기 공부

# mlp for multi-label classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import RepeatedKFold
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score

# get the dataset
def get_dataset():
	X, y = make_multilabel_classification(n_samples=1000, n_features=10, n_classes=3, n_labels=2, random_state=1)
	return X, y

# get the model
def get_model(n_inputs, n_outputs):
	model = Sequential()
	model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(n_outputs, activation='sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam')
	return model

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
		model.fit(X_train, y_train, verbose=0, epochs=100)
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

# load dataset
X, y = get_dataset()
# evaluate model
results = evaluate_model(X, y)
# summarize performance
print('Accuracy: %.3f (%.3f)' % (mean(results), std(results)))



# # 저장하고 불러와서 예측 할 때 ============================================

# # use mlp for prediction on multi-label classification
# from numpy import asarray
# from sklearn.datasets import make_multilabel_classification
# from keras.models import Sequential
# from keras.layers import Dense
 
# # get the dataset
# def get_dataset():
# 	X, y = make_multilabel_classification(n_samples=1000, n_features=10, n_classes=3, n_labels=2, random_state=1)
# 	return X, y
 
# # get the model
# def get_model(n_inputs, n_outputs):
# 	model = Sequential()
# 	model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
# 	model.add(Dense(n_outputs, activation='sigmoid'))
# 	model.compile(loss='binary_crossentropy', optimizer='adam')
# 	return model
 
# # load dataset
# X, y = get_dataset()
# n_inputs, n_outputs = X.shape[1], y.shape[1]
# # get model
# model = get_model(n_inputs, n_outputs)
# # fit the model on all data
# model.fit(X, y, verbose=0, epochs=100)
# # make a prediction for new data
# row = [3, 3, 6, 7, 8, 2, 11, 11, 1, 3]
# newX = asarray([row])
# yhat = model.predict(newX)
# print('Predicted: %s' % yhat[0])