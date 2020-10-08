import sys
# print("python version {}".format(sys.version))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# import mglearn
import sklearn

# import scipy as sp

# eye=np.eye(4)
# print(eye)
# print(np.arange(4))

# x=np.linspace(-3,3,10)
# y=np.sin(x)
# plt.plot(x,y, marker='x')
# plt.show()

from sklearn.datasets import load_iris

iris_dataset = load_iris()

print(iris_dataset.keys())
print(iris_dataset['DESCR'])
print(iris_dataset['target_names'])
print(type(iris_dataset['target_names']))
print(iris_dataset['data'].shape)
print(iris_dataset['target'].shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
X_train[:5]
iris_dataset.keys()
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
X_new = np.array([[5, 2.9, 1.0, 0.2]])
X_new_false = np.array([5, 2.9, 1.0, 0.2])
prediction = knn.predict(X_new)
prediction[0]
# input should be 2d ndarray, not rank1 1d array like of size (4,)

prediction2 = knn.predict(X_new_false.reshape(1, -1))
prediction2[0]
print("input:{},  target name ->{}".format(X_new, iris_dataset['target_names'][prediction]))

y_pred = knn.predict(X_test)
y_pred
y_test

print('in sample accuracy {:.5f}'.format(np.mean(knn.predict(X_train) == y_train)))
print('out of sample accuracy {:.5f}'.format(np.mean(y_pred == y_test)))
knn.score(X_test, y_test)
type(knn)
