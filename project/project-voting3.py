import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# 1. 데이터
path = './data/credit_card_prediction/'
datasets = pd.read_csv(path + 'train.csv')

x = datasets[['ID', 'Gender', 'Age', 'Region_Code', 'Occupation', 'Channel_Code',
       'Vintage', 'Credit_Product', 'Avg_Account_Balance', 'Is_Active']]
y = datasets[['Is_Lead']]

# 필요없는 칼럼 제거
x = x.drop(['Age', 'Vintage'], axis=1)
y = y.iloc[x.index]  # x에서 제거한 인덱스에 해당하는 y 값도 제거

# NaN 값 처리
x = x.fillna(x.mode())

# LabelEncoder
ob_col = list(x.dtypes[x.dtypes=='object'].index) # object 컬럼 리스트
for col in ob_col:
    x[col] = LabelEncoder().fit_transform(x[col].values)

# train 데이터와 test 데이터 분할
train_data_ratio = 0.8
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1-train_data_ratio, random_state=123)

# 모델 정의
nc_model = KNeighborsRegressor(n_neighbors=7)
rf_model = RandomForestRegressor()
dt_model = DecisionTreeRegressor()

model = VotingRegressor(
    estimators=[
                ('nc_model', nc_model),
                ('rf_model', rf_model),
                ('dt_model', dt_model)],
    n_jobs=-1,
)

# 모델 훈련
model.fit(x_train, y_train)

# 모델 사용하여 테스트 세트 예측 및 평가
classifiers = [nc_model, rf_model, dt_model]

result = model.score(x_test, y_test)
print('Voting 결과: ', result)