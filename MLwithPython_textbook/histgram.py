import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick
from matplotlib import font_manager, rc

font_name = font_manager.FontProperties(fname="d:/hangul/NanumGothic.ttf").get_name()
rc('font', family=font_name)

# data = [8,12,15,17,18,18.5]
# perc = np.linspace(0,100,len(data))

fig = plt.figure(1, (7, 4))
ax = fig.add_subplot(1, 1, 1)

# ax.plot(perc, data)

fmt = '%.3f%%'  # Format you want the ticks, e.g. '40%'
xticks = mtick.FormatStrFormatter(fmt)
ax.xaxis.set_major_formatter(xticks)

# df=pd.read_csv("D:/dev/cpp/DERIV_SHB2/SingleOption/PL01161441.csv")
# df = pd.read_csv("total.csv")
df = pd.read_csv("PL_flat2.csv")
df_els = df['PL']
df_put = df['PL_local']
# df_put = df['maxloss flat']

for i in df_els.index:
    # df_els.set_value(i, df_els.get_value(i) * 100)
    # df_put.set_value(i, df_put.get_value(i) * 100)
    pass

for i, v in enumerate(df_els):
    df_els[i] = v * 100

for i, v in enumerate(df_put):
    df_put[i] = v * 100

plt.hist(df_els, bins=100, range=(-3, 3), alpha=0.5)
plt.hist(df_put, bins=100, range=(-3, 3), alpha=0.5)
plt.xlabel("손익(%)", fontsize=12)
plt.ylabel("횟수", fontsize=12)
# plt.legend(labels=['최종손익'])
# plt.hist(df['PL'])

# plt.title("Local 변동성 헤지 운용시 \n모형차이에 따른 운용기간동안의 최대 평가손실(노셔널 100%)")
plt.title("Flat vol 모형 vs Local vol 모형- 최종손익(노셔널 100%)")

plt.show()
# print(df['PL'].mean())
