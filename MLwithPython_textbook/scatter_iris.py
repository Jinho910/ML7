# code from
# https://datascience.stackexchange.com/questions/33780/scatter-plot-for-binary-class-dataset-with-two-features-in-python

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
features = iris.data.T
featuresT = iris.data
features.shape
features
featuresT
type(features)
plt.scatter(features[0], features[1], alpha=0.2,
            s=100 * features[1], c=iris.target, cmap='viridis')
plt.scatter()
# plt.scatter(features[0], features[1], alpha=0.7,
#             s=100*features[1], c=iris.target)

plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()
