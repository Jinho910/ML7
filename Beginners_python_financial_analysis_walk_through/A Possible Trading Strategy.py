# replication of https://towardsdatascience.com/a-possible-trading-strategy-technical-analysis-with-python-ee1168b5f117
import datetime
import yfinance as yf
import numpy as np
import pandas as pd

stocks = ['^IXIC']  # NASDAQ

start = datetime.datetime(2017, 1, 1)
end = datetime.datetime(2019, 1, 1)

data = yf.download(tickers=stocks, start=start, end=end, auto_adjust=True)
data.head()

rsi_period = 14
chg = data['Close'].diff(1)
gain = chg.mask(chg < 0, 0)  # mask negative value as 0
loss = chg.mask(chg > 0, 0)
data['gain'] = gain
data['loss'] = loss

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.ewm.html
avg_gain = gain.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
avg_loss = loss.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
data['avg_gain'] = avg_gain
data['avg_loss'] = avg_loss

rs = abs(avg_gain / avg_loss)
rsi = 100 - (100 / (1 + rs))
data['rsi'] = rsi

data.plot(y=['Close', 'rsi'])
data10 = data[:100]
data10.plot(y=['Close', 'rsi'])
# TODO subplot


nasdaq = data['Close'].pct_change()  # TODO pct_change with log return ?
nasdaq.count()  # count except NaN
mean_nasdaq = nasdaq.sum() / nasdaq.count()
sqd_nasdaq = nasdaq.apply(lambda x: (x - mean_nasdaq))
ssqd = sqd_nasdaq.sum()
var_nasdaq = ssqd / (nasdaq.count() - 1)
v_nasdaq = nasdaq.var()
np.sqrt(v_nasdaq)

# TODO make class for stock analysis
