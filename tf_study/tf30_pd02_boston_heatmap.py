import numpy as np
import pandas as pd

#1. 데이터
path = './data/boston/'
datasets = pd.read_csv(path + 'Boston_house.csv')

print(datasets.columns)
print(datasets.head(7))

x = datasets[['AGE', 'B', 'RM', 'CRIM', 'DIS', 'INDUS', 'LSTAT', 'NOX', 'PTRATIO',
       'RAD', 'ZN', 'TAX', 'CHAS']]
y = datasets[['Target']]

print(x.shape, y.shape) # (506, 13) (506, 1)

# 상관계수 히트맵(heatmap)
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale = 1.2)
sns.set(rc = {'figure.figsize':(12,9)})
sns.heatmap (
       data=datasets.corr(), #corr()는 상관관계
       square=True,
       annot=True,
       cbar=True
)

plt.show()
