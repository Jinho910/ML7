# followed from https://levelup.gitconnected.com/scikit-learn-python-6-useful-tricks-for-data-scientists-1a0a502a6aa3

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer

print(3)

X, y = make_classification(n_samples=10, n_features=4, n_classes=2, random_state=123)

X = pd.DataFrame(X, columns=['Feature_1', 'Feature_2', 'Feature_3', 'Feature_4'])
y = pd.DataFrame(y, columns=['Label'])

X.iloc[1, 2] = float('NaN')

imputer_simple = SimpleImputer()
X_imputed = pd.DataFrame(imputer_simple.fit_transform(X))

imputer_KNN = KNNImputer(n_neighbors=2, weights='uniform')
X_imputed_KNN = pd.DataFrame(imputer_KNN.fit_transform(X))
