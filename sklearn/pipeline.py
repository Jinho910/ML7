# followed from https://levelup.gitconnected.com/scikit-learn-python-6-useful-tricks-for-data-scientists-1a0a502a6aa3


import joblib
from sklearn.datasets import make_classification
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import pandas as pd

X, y = make_classification(n_samples=25, n_features=4, n_classes=2, random_state=123)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

imputer = SimpleImputer()
clf = LogisticRegression()

pipe = make_pipeline(imputer, clf)

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

result = pd.DataFrame({'Prediction': y_pred, 'True': y_test})
print(result)

joblib.dump(pipe, 'simple_impute_logistic.joblib')

new_pipe = joblib.load('simple_impute_logistic.joblib')
y_pred_new = new_pipe.predict(X_test)
