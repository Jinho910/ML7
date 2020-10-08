# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 22:41:41 2019

@author: Administrator
"""


import mglearn
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
X,y=mglearn.datasets.make_forge()
X
mglearn.discrete_scatter(X[:,0], X[:,1],y)
plt.legend(["class 0", "class 1"],loc=4)
plt.xlabel(" first feature")
plt.show()


fig, axes=plt.subplots(1,3,figsize=(10,3))
#axes
for n_neighbors, ax in zip([1,3,9],axes):
    #fit methods는 self 객체를 반환하므로 아래와 같이 한줄로 표현 가능
    clf=KNeighborsClassifier(n_neighbors=n_neighbors).fit(X,y)
    mglearn.plots.plot_2d_separator(clf,X,fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:,0],X[:,1],y,ax=ax)
    ax.set_title("{} neighbors".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend(loc=4)
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)  #output 순서에 주의


print(X_train)
print(y_train)
from sklearn.neighbors import KNeighborsClassifier
#module : sklearn.neighbors 
#class : KNeighborsClassifier

clf=KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)

print("테스트 세트 예측: {}".format(clf.predict(X_test)))
print("정확도 : {:.2f}".format(clf.score(X_test,y_test)))


from sklearn.datasets import load_breast_cancer
X,y=mglearn.datasets.make_wave(n_samples=40)
X.shape # input var X should be 2d matrix,item in each row
plt.plot(X,y,'o')
plt.xlabel(' 특성 ')
plt.ylabel('target')
plt.show()



cancer=load_breast_cancer()
print("cancer.keys(): \n{}".format(cancer.keys()))
cancer.data.shape
type(cancer.data)
cancer.target_names

import numpy as np 
cancer.target.shape
np.bincount(cancer.target)
dic={n:v for n,v in zip(cancer.target_names, np.bincount(cancer.target))}
dic
print("클래스별 샘플 개수:\n{}".format({n:v for n,v in zip(cancer.target_names, np.bincount(cancer.target))}))
print("특성이름:\n{}".format(cancer.feature_names))
print(cancer.target_names)

from sklearn.datasets import load_boston
boston=load_boston()
print("데이터형태:\n{}".format(boston.data.shape))
X,y=mglearn.datasets.load_extended_boston()
print("X.shpae: {}".format(X.shape))

mglearn.plots.plot_knn_classification(n_neighbors=3)
plt.show()

from sklearn.model_selection import train_test_split
X,y=mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=0)

X_train.shape
from sklearn.neighbors import KNeighborsClassifier
clf3=KNeighborsClassifier(n_neighbors=3)
clf2=KNeighborsClassifier(n_neighbors=2)
clf1=KNeighborsClassifier(n_neighbors=1)
clf9=KNeighborsClassifier(n_neighbors=9)


clf1.fit(X, y)
clf2.fit(X, y)
clf3.fit(X, y)
clf9.fit(X, y)

print('1N {:.5f}'.format(clf1.score(X,y)))
print('2N {:.5f}'.format(clf2.score(X,y)))
print('3N {:.5f}'.format(clf3.score(X,y)))
print('9N {:.5f}'.format(clf9.score(X,y)))

cancer=load_breast_cancer()
ran=50
X_train, X_test, y_train, y_test=train_test_split(cancer.data, cancer.target, random_state=ran)
training_accuracy=[]
test_accuracy=[]

neighbors_setting=range(1,11)

for n_neighbors in neighbors_setting:
    clf=KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    training_accuracy.append(clf.score(X_train, y_train))
    test_accuracy.append(clf.score(X_test, y_test))

plt.plot(neighbors_setting, training_accuracy, label="train")
plt.plot(neighbors_setting, test_accuracy, label="test")
plt.ylabel("accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

mglearn.plots.plot_knn_regression(n_neighbors=10)
plt.show()

from sklearn.neighbors import KNeighborsRegressor
X,y=mglearn.datasets.make_wave(n_samples=40)
X_train, X_test, y_train, y_test=train_test_split(X,y, random_state=0)
reg=KNeighborsRegressor(n_neighbors=3)
reg.fit(X_train, y_train)
print("R^2 measure {:.2f}".format(reg.score(X_test,y_test)))
fig, axes=plt.subplots(1,3,figsize=(15,4))

line=np.linspace(-3,3,1000).reshape(-1,1) # (nx1 2d array로 변환)
for n_neighbors, ax in zip([1,3,9],axes):
    reg=KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train, y_train) 
    ax.plot(line, reg.predict(line))
    ax.plot(X_train, y_train , "^", c=mglearn.cm2(0), markersize=8)
    ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)
    ax.set_title("{} neighbors train score {:.2f}, test score{:.2f}".format(n_neighbors,reg.score(X_train, y_train),reg.score(X_test, y_test)))
    ax.set_xlabel("feature")
    ax.set_ylabel("target")
axes[0].legend(["model prediction", "train data/target", "test data/target"], loc="best")
plt.show()

mglearn.plots.plot_linear_regression_wave()
plt.show()

from sklearn.linear_model import LinearRegression
X,y=mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=42)
lr=LinearRegression()
lr.fit(X_train, y_train)
lr.coef_
lr.intercept_
line=np.linspace(-3,3,1000).reshape(-1,1)
plt.plot(line, lr.predict(line))
plt.plot(X_train, y_train, "^")
plt.plot(X_test, y_test, "*")
plt.show()

from sklearn.linear_model import Ridge
X,y=mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test=train_test_split(X,y,random_state=0)

lr=LinearRegression().fit(X_train, y_train)
lr.score(X_test, y_test)
lr.score(X_train, y_train)

boston=load_boston()
boston.keys()
X_origin=boston['data']
y_origin=boston['target']
X_origin_train, X_origin_test, y_origin_train, y_origin_test=train_test_split(X_origin, y_origin)

lr_origin=LinearRegression().fit(X_origin_train, y_origin_train)
lr_origin.score(X_origin_test, y_origin_test)
lr_origin.score(X_origin_train, y_origin_train)

ridge=Ridge(alpha=20).fit(X_train, y_train)
#ridge.fit(X_train, y_train)
ridge.score(X_train, y_train)
ridge.score(X_test,y_test)

score=[]
score=np.empty(0)
score=np.append(score, 1.2)
score= np.append(score, 2.3)

for n in list(range(0,10)):
    ridge=Ridge(alpha=n).fit(X_train, y_train)
    score.append(ridge.score(X_test,y_test))
    print(ridge.score(X_test, y_test))

score=np.array([])
alphas=np.arange(0,3,0.1)

for alpha in alphas:
    ridge=Ridge(alpha=alpha).fit(X_train, y_train)
    score_added=ridge.score(X_test, y_test)
    print(score_added)
    score=np.append(score,score_added)

alphas=np.arange(0,10,1)
for alpha in alphas:
    ridge_origin=Ridge(alpha=alpha).fit(X_origin_train, y_origin_train)
    score_added=ridge_origin.score(X_origin_test, y_origin_test)
    print(score_added)
    score=np.append(score,score_added)



plt.plot(alphas, score, "o-")
plt.xlabel('alpha')
plt.ylabel('score')
plt.show()

ridge10=Ridge(alpha=10).fit(X_train, y_train)
ridge01=Ridge(alpha-0.1).fit(X_train, y_train)
ridge =Ridge().fit(X_train, y_train)
plt.plot(ridge10.coef_, '^', label="ridge alpha10")
plt.plot(ridge.coef_, 's', label="ridge alpha=1(default)")
# plt.plot(ridge01.coef_, 'v', label="ridge alpha 0.1")
np.sum(ridge.coef_!=0) #0이 아닌 계수개수
plt.plot(lr.coef_, 'o', label="Linear regeression ")
plt.xlabel("coefficients")
plt.ylabel("amplitude")
xlims=plt.xlim()
plt.hlines(0, xlims[0], xlims[1])
plt.xlim(xlims)
plt.ylim(-25, 25)
plt.legend()
plt.show()

from sklearn.linear_model import Lasso
lasso=Lasso().fit(X_train, y_train)
np.sum(lasso.coef_!=0)
lasso001=Lasso(alpha=0.001, max_iter=100000).fit(X_train, y_train)
np.sum(lasso001.coef_!=0)
lasso001.score(X_test, y_test)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

X,y=mglearn.datasets.make_forge()
fig, axes=plt.subplots(1,1)
clf=LogisticRegression().fit(X,y)

plt.scatter(X[:,0], X[:,1], alpha=0.2, c=y)
clf.coef_[0,1]
clf.intercept_
plt.show()

a=np.array([[1,2],[3,4]])
b=np.array([[5,6]])
c=np.matmul(a,b.T)
