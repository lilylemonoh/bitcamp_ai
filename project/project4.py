import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# 1. 데이터
path = '/home/ncp/workspace/_data/credit_card_prediction/credit_card_prediction/'
datasets = pd.read_csv(path + 'train.csv')

print(datasets.columns)

print('******Labeling 전 데이터*****')
print(datasets.head(11))

# 문자를 숫자로 변경 (LabelEncoder)
df = pd.DataFrame(datasets)

ob_col = list(df.dtypes[df.dtypes=='object'].index) # object 컬럼 리스트

# NaN 처리
df['Credit_Product'].fillna('Unknown', inplace=True)


for col in ob_col:
    df[col] = LabelEncoder().fit_transform(df[col].values)
    
print('******Labeling 후 데이터*****')
print(datasets.head(11))
    
x = datasets[['ID', 'Gender', 'Age', 'Region_Code', 'Occupation', 'Channel_Code',
       'Vintage', 'Credit_Product', 'Avg_Account_Balance', 'Is_Active']]
y = datasets[['Is_Lead']]

print(x.shape) # (245725, 10)
print(y.shape) # (245725, 1)