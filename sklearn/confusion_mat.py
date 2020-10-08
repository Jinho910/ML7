# followed from https://levelup.gitconnected.com/scikit-learn-python-6-useful-tricks-for-data-scientists-1a0a502a6aa3

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = make_classification(n_samples=1000, n_features=4, n_classes=2, random_state=123)
y.shape
X.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
clf = LogisticRegression()
clf.fit(X_train, y_train)
confmat = plot_confusion_matrix(clf, X_test, y_test, cmap="Blues")
plt.show()

# True positive : 예측 1(positive) 맞춤(true)
# True neg : 예측 0(neg),  맞춤(true)
# False pos : 예측 1(pos) , 틀림(false) 실제는 neg
# Flase neg: 예측 0(neg), 틀림(false) 실제는 pos  => 병원에서 중요
