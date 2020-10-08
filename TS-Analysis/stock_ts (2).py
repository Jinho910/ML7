import numpy as np
import pandas_datareader.data as web
import pandas as pd
from tsplot import tsplot
import matplotlib.pyplot as plt

start = '2019-01-01'
end = '2020-07-23'
get_px = lambda x: web.DataReader(x, 'naver', start=start, end=end)['Close']

symbols = ['005930', '035420', '035720']
data = pd.DataFrame({sym: get_px(sym) for sym in symbols})  # index가 다르면 어떻게 되나?
# convert objects typt to float
data = data.astype(float)
# log return
lr = np.log(data / data.shift(1)).dropna()
# _ = tsplot.tsplot(data['005930'], lags=30)
# data.plot(kind='line' x='')
print(data)

# ax = plt.gca()
# data.plot(kind='line', y='005930', ax=ax)
# plt.show()
# print(np.diff(data['005930']))
# _ = tsplot(np.diff(data['005930'])) #very like white noise but with heavy tail.

# random walk for comparison
n_sample = 1000
x = np.zeros(n_sample)
w = np.random.normal(size=n_sample)
for i in range(n_sample):
    if i == 0:
        x[i] = w[0]
    else:
        x[i] = x[i - 1] + w[i]
# plt.plot(x)
# plt.show()
diff_x = np.diff(x)
# tsplot(diff_x, lags=30)

# linear time series
n_sample = 100
w = np.random.randn(n_sample)
y = np.zeros(n_sample)
b0 = -50
b1 = 2.5
for t in range(len(w)):
    y[t] = b0 + b1 * t + w[t]

_ = tsplot(y, lags=30)
