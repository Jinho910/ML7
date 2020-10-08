# followed from https://levelup.gitconnected.com/scikit-learn-python-6-useful-tricks-for-data-scientists-1a0a502a6aa3

from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

X, y = make_classification(n_samples=50, n_features=4, n_classes=2, random_state=123)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

clf = DecisionTreeClassifier()
model = clf.fit(X_train, y_train)
y_predic = model.predict(X_test)
clf.score(X_test, y_test)

confmat = plot_confusion_matrix(clf, X_test, y_test, cmap="Blues");
plt.show()

plot_tree(clf, filled=True);
plt.show()
