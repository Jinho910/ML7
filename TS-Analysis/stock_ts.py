import pandas_datareader.data as web
import numpy as np
import pandas as pd

start = "2019-01-01"
end = "2020-07-24"
df_stock = web.DataReader("005930", "naver", start=start, end=end)
print(df_stock["Close"].tail(5))
df_stock["Close"].shape
shifted_stock = df_stock.shift(1)  # 인덱스는 그대로 두고 content만 민다. 첫 행은 NaN
shifted_stock.shape
print(f'shifted stock \n{shifted_stock["Close"].tail(5)}')

get_px = lambda x: web.DataReader(x, "naver", start=start, end=end)
data = pd.DataFrame(get_px("005930"))


print(data.tail(5))
print(f"data\n{data.tail(5)}")


end = "2015-01-01"
start = "2007-01-01"
get_px = lambda x: web.DataReader(x, "yahoo", start=start, end=end)["Adj Close"]

symbols = ["SPY", "TLT", "MSFT"]
# raw adjusted close prices
data = pd.DataFrame({sym: get_px(sym) for sym in symbols})
data.head(5)
data.shift(1).head(5)
lrets = np.log(data / data.shift(1)).dropna()
lrets.head(5)
# log returns