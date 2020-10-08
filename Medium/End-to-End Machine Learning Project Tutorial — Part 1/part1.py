import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

cols = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
        'Acceleration', 'Model Year', 'Origin']

data = pd.read_csv('D:/dev/python/ML/Medium/End-to-End Machine Learning Project Tutorial — Part 1/auto-mpg.data',
                   names=cols, na_values="?",
                   comment='\t', sep=" ", skipinitialspace=True)

data.head()
data.info()
data.describe()
data.isnull().sum()  # 각 attr에 대해서 null이 몇개인지, Horsepower에 null값이 있음
# missing value처리
data['Horsepower'].describe()
sns.boxplot(x=data['Horsepower'])
# imputing the values with median
median = data['Horsepower'].median()
data['Horsepower'] = data['Horsepower'].fillna(median)
data['Horsepower'].isnull().sum()
data['Cylinders'].hist()
data['Cylinders'].value_counts() / len(data)  # number of unique value
data['Origin'].value_counts()

# plot for correlation
sns.pairplot(data[["MPG", "Cylinders", "Displacement", 'Weight', 'Horsepower']],
             diag_kind="kde")
