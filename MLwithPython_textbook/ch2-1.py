import mglearn
import matplotlib.pyplot as plt

X, y = mglearn.datasets.make_forge()
y
import numpy as np

# X[:,0]
# X[:,0].shape
# X.shape
# mglearn.discrete_scatter(X[:,0], X[:,1],y)
# plt.scatter(X[:,0], X[:,1])
# plt.show()
# N = 100
N = 5
r0 = 0.6
x = 0.9 * np.random.rand(N)
y = 0.9 * np.random.rand(N)
area = (20 * np.random.rand(N)) ** 2
# area
# c = np.sqrt(area)
r = np.sqrt(x ** 2 + y ** 2)
area1 = np.ma.masked_where(r < r0, area)

y1 = np.ma.masked_where(y >= 1, y)
y0 = np.ma.masked_where(y == 0, y)

# '#7f7f7f', '#bcbd22',
# type(area1)
# area2 = np.ma.masked_where(r >= r0, area)
plt.scatter(X[:, 0], X[:, 1], s=y1 * 20 ** 2, marker='^')

# plt.scatter(x, y, s=area2, marker='o', c=c)
# Show the boundary between the regions:
# theta = np.arange(0, np.pi / 2, 0.01)
# plt.plot(r0 * np.cos(theta), r0 * np.sin(theta))

plt.show()
