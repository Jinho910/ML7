from datetime import datetime

import pandas as pd
import pandas_datareader as pdr

start_date = '2018-01-01'
print(start_date)

end_date = datetime.today()
end_date = end_date.strftime("%y-%m-%d")
print(end_date)
tickers = ['SPY', "AAL", "ZM", "NFLX", "FB"]
each_df = {}
for ticker in tickers:
    each_df[tickers] = pdr.DataReader(ticker, 'yahoo', start_date, end_date)

stocks = pd.concat(each_df, axis=1, keys=tickers)  # axis=1 -> stack horizontally
