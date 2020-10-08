from tsplot import tsplot
import numpy as np
import pandas_datareader.data as web


np.random.seed(1)
n_samples = 1000
w = np.random.normal(size=n_samples)
x = np.zeros(n_samples)
for t in range(n_samples):
    if t == 0:
        x[t] = w[t]
    else:
        x[t] = x[t - 1] + w[t]

# splot(x, lags=30) #random walk is non-stationary

diff_x = np.diff(x)
tsplot(diff_x, lags=30)  # increment of random walk is random(white noise)
start = "2019-01-01"
end = "2020-07-22"
df_stock = web.DataReader("005930", "naver", start=start, end=end)

print(df_stock.head())
