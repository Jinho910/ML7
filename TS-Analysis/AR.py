# http://www.blackarbs.com/blog/time-series-analysis-in-python-linear-models-to-garch/11/1/2016
import os
import sys
import pandas as pd
import pandas_datareader.data as web

df = web.DataReader("005930", "naver", start="2019-09-10", end="2020-07-21")
df.head()
print(df)
