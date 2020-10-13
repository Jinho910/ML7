import yfinance as yf
import pandas as pd
from pandas_datareader import data as pdr

# yf.pdr_override()
# data = yf.download("SPY AAPL", start="2018-01-01", end='2020-01-01')
# data = pdr.get_data_yahoo("SPY", start='2018-01-01', end='2020-01-01')

stocks = yf.download(tickers='SPY AAPL', period='ytd', interval='1d', group_by='ticker', auto_adjust=True)
amz = yf.download(tickers='AMZN', start='2020-08-01', end='2020-09-28', auto_adjust=True)
amz.head()
amz.describe()
amz.dtypes
amz.columns
amz.describe(include="float")

# masking close price > 3000
mask_closeprice = amz.Close > 3000
mask_closeprice3500 = amz['Close'] > 3500
over3000 = amz.loc[mask_closeprice]
over3000
over3500 = amz.loc[mask_closeprice3500]
mask_close_over3200 = amz.Close > 3200
mask_vol_over4000000 = amz.Volume > 4000000

close_over3200_vol_over4000000 = amz.loc[mask_close_over3200 & mask_vol_over4000000]

# amz.plot(x='Date', y='Close')
amz.plot(y='Close', rot=90)
amz.plot(y='Close', rot=90, title='AMZN Close Price')
# amz.plot(y='Close', kind='scatter', rot=90, title='Amazon close price')
amz.plot(y="Close", rot=90, kind='hist')
amz.Close
