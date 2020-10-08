# a=['hello', 'python', 'nice', 'to', 'meet', 'you']
# for i in range(len(a)):
#     print(i, a[i])


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick

# data = [8,12,15,17,18,18.5]
# perc = np.linspace(0,100,len(data))

fig = plt.figure(1, (7,4))
ax = fig.add_subplot(1,1,1)

# ax.plot(perc, data)

fmt = '%.2f%%' # Format you want the ticks, e.g. '40%'
xticks = mtick.FormatStrFormatter(fmt)
ax.xaxis.set_major_formatter(xticks)


df=pd.read_csv("D:/dev/cpp/DERIV_SHB2/SingleOption/PL02051128.csv")
d=df['PL']

# for i in d.index:
#     val=d.get_value(i)
#     d.set_value(i,val*100)


plt.hist(d, bins=200, range=(-.05,.05))
# plt.hist(df['PL'])
# plt.hist(df['PL'])

plt.title("PL distribution(% of notional)")
plt.show()
print(df['PL'].mean())