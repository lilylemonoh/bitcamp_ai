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

x['NaN'] = np.nan

print(x)
print(x.info())

#NaN 값 처리
# 1. 0으로 채우기
x = x.fillna(0)

print(x)
print(x.info())


# #2. 평균(mean)으로 채우기
# x = x.fillna(x.mean())

# #3. 중간값(mode), 최소값(Min), 최대값(Max)로 채우기
# x = x.fillna(x.mode())
# x = x.fillna(x.min())
# x = x.fillna(x.max())



# #4. 해당 데이터의 앞 데이터 값으로 채우기
# x = x.fillna(method='ffill')

# #5. 해당 데이터의 뒤 데이터 값으로 채우기
# x = x.fillna(method='bfill')

# print(x)
# print(x.info())