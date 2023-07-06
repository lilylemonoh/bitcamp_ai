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

# 문자를 숫자로 변경(LabelEncoder)
from sklearn.preprocessing import LabelEncoder
ob_col = list(x.dtypes[x.dtypes == 'object'].index) # object 컬럼 리스트
for col in ob_col:
       x[col] = LabelEncoder().fit_transform(x[col].values)
       
       

       
       #히트맵, 피처임포터트