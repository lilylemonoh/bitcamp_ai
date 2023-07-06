import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error

# 1. 데이터
path = './data/credit_card_prediction/'
datasets = pd.read_csv(path + 'train.csv')

x = datasets[['ID', 'Gender', 'Age', 'Region_Code', 'Occupation', 'Channel_Code',
       'Vintage', 'Credit_Product', 'Avg_Account_Balance', 'Is_Active']]
y = datasets[['Is_Lead']]

# 필요없는 칼럼 제거
x = x.drop(['Age', 'Vintage'], axis=1)

# NaN 값 처리
x = x.fillna(x.mode())

# LabelEncoder
ob_col = list(x.dtypes[x.dtypes=='object'].index) # object 컬럼 리스트
for col in ob_col:
    x[col] = LabelEncoder().fit_transform(x[col].values)

# train 데이터와 test 데이터 분할
train_data_ratio = 0.8
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1-train_data_ratio, random_state=123)

# 결과를 출력하여 분할이 성공적으로 완료되었는지 확인합니다.
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


nc_model = KNeighborsRegressor(n_neighbors=7)
rf_model = RandomForestRegressor()
dt_model = DecisionTreeRegressor()
cat= CatBoostRegressor()
xgb = XGBRegressor()
lgb = LGBMRegressor()


model = VotingRegressor(
    estimators=[
                ('nc_model', nc_model),
                ('rf_model :', rf_model),
                ('dt_model :', dt_model),
                ('cat', cat),
                ('xgb :', xgb),
                ('lgb :', lgb)],
    n_jobs=-1,
)


# 모델을 훈련합니다.
model.fit(x_train, y_train)

# 모델을 사용하여 테스트 세트를 예측하고 정확도를 출력합니다.

classifiers = [nc_model, rf_model, dt_model, cat, lgb, xgb]

result = model.score(x_test, y_test)
print('Voting 결과: ', result)

