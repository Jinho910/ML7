import yfinance as yf
import datetime
import pandas as pd
# from pandas_datareader import data as pdr

# yf.pdr_override()
# data = yf.download("SPY AAPL", start="2018-01-01", end='2020-01-01')
# data = pdr.get_data_yahoo("SPY", start='2018-01-01', end='2020-01-01')
start_date='2018-01-01'
end_date=datetime.date.today().strftime("%Y-%m-%d")
tickers=["SPY", "AAL", "ZM", "NFLX", "FB"]
stocks = yf.download(tickers=["SPY", "AAL", "ZM", "NFLX", "FB"], period='ytd', interval='1d', group_by='ticker', auto_adjust=True)
stocks2=yf.download(tickers=tickers, start=start_date, end=end_date,auto_adjust=True)

each_df={}
for ticker in tickers:
    each_df[ticker]=yf.download(tickers=ticker,start=start_date, end=end_date, auto_adjust=True)

each_df['SPY']
#dictionary of dataframe -> multi-index dataframe with key
stocks3=pd.concat(each_df, axis=1, keys=tickers)
stocks3.columns

stocks.columns
stocks.shape
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
amz.columns
stocks.info()
#set floating decimal
pd.options.display.float_format='${:,.2f}'.format
amz.shape
type(amz)
