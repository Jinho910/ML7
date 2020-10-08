# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 13:54:49 2019

@author: Administrator
"""
# %matplotlib inline

import sys

print("Python 버전:{}".format(sys.version))
# from IPython.display import display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

print("numpy version {}".format(np.__version__))
import scipy as sp

print("scipy version {}".format(sp.__version__))

from scipy import sparse

eye = np.eye(4)
print(eye)
print(eye.shape, type(eye))
sparse_matrix = sparse.csr_matrix(eye)
print(sparse_matrix)

data = np.ones(4)
print(data)
print(data.shape)
row_indices = np.arange(4)
print(row_indices)
print(row_indices.shape)

x = np.linspace(-3, 3, 10)  # element  개수 =10 개임

print(x)
print(x.shape)
y = np.sin(x)
print(y.shape)
plt.plot(x, y, marker='x')
# plt.show()

# data frame

data = {'Name': ["정현", "한나", "도언", "유주"],
        'Location': ["광주", "중계동", "중앙", "하이츠"],
        'Age': [43, 35, 7, 4]}

from sklearn.datasets import load_iris

iris_dataset = load_iris()
# type of iris_dataset:utils.bunch, a bunch object of sklearn.utils module

print("iris_dataset key \n{}".format(iris_dataset.keys()))
print(iris_dataset.keys())

print(iris_dataset['DESCR'])
type(iris_dataset['DESCR'])

type(iris_dataset)
print(type(iris_dataset['target_names']))
print(iris_dataset['target_names'][:5])
type(iris_dataset['data'])

print(iris_dataset['data'].shape)
print(iris_dataset['data'][:5])  # [0,5)

print(iris_dataset['data'][:5, 1:3])  # [0,5)x[1,3)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

type(X_train)

# scatter_matrix - data visualisation
# iris_dataframe=pd.DataFrame(X_train,columns=iris_dataset.feature_names)
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset['feature_names'])

print(iris_dataset['feature_names'])

# K nearest neighber 알고리즘 사용
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

X_new = np.array([[5, 2.9, 1, 0.2]])
print(X_new.shape)  # scikit-learn은 항상 데이터가 2차원 nd array로 예상함
X_new_false = np.array([5, 2.9, 1, 0.2])  # this is not a 2d array

# prediction
prediction = knn.predict(X_new)
# prediction_wrong_datatype=knn.predict(X_new_false)
# input error

prediction2 = knn.predict(X_new_false.reshape(1, -1))  # 2D 1xn(row array), .reshape(-1,1) returns nx1 column array
prediction2

print(str(X_new) + "-> 예측 {}".format(prediction))
# iris_target_names=iris_dataset['target_name']

print("예측한 타겟의 이름 {}".format(iris_dataset['target_names'][prediction]))

# 테스트 셋으로 모델 평가하기
y_pred = knn.predict(X_test)
y_pred
y_test

# 테스트의 정확도 보기
# True==1, False==0 으로 취급하여 np.mean 하는 경우 accuracy를 표현함
print("테스트의 정확도(accuracy) {:.5f}".format(np.mean(y_pred == y_test)))

y_pred == y_test

np.mean([True, True])
# score 멤버함수가 위를 대신함
print("테스트의 정확도 {:2f}".format(knn.score(X_test, y_test)))
