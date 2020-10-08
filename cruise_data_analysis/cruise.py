import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("cruise_ship_info.csv")

cols=['Age', 'Tonnage', 'passengers', 'length',
       'cabins', 'passenger_density', 'crew']


from sklearn.preprocessing import StandardScaler

stdsc=StandardScaler()

X_std=stdsc.fit_transform(df[cols].iloc[:,range(0,7)].values)
cov_mat=np.cov(X_std.T)
print(type(cov_mat))
print(cov_mat)

plt.figure(figsize=(10,10))
sns.set(font_scale=1.5)
sns.heatmap(cov_mat,cbar=True, annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 12},
                 yticklabels=cols,
                 xticklabels=cols)
plt.title('Covariance matrix showing correlation coefficients')
plt.tight_layout()
# plt.show()

cols_selected=['Tonnage', 'passengers', 'length', 'cabins', 'crew']
print(df[cols_selected].head())

X=df[cols_selected].iloc[:,0:4].values
y=df[cols_selected]['crew'].values


