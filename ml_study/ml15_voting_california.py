from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, LinearSVC
import numpy as np
from sklearn.ensemble import VotingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data # x = datasets['data]와 동일함
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=72,
    shuffle=True
)

# Scaler (정규화) 적용
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#kfold
n_splits = 5
random_state = 62
kfold = StratifiedKFold(  #kFold=회귀모델 / StratifiedKFold = 분류모델
    n_splits=n_splits,
    random_state=random_state,
    shuffle=True
)

#2. 모델
nc_model = KNeighborsRegressor(n_neighbors=7)
rf_model = RandomForestRegressor()
dt_model = DecisionTreeRegressor()

cat= CatBoostRegressor()
xgb = XGBRegressor()
lgb = LGBMRegressor(learning_rate=0.3,
                    max_depth=10,
                    random_state=1004)


model = VotingRegressor(
    estimators=[
                # ('nc_model', nc_model),
                # ('rf_model :', rf_model),
                # ('dt_model :', dt_model),
                ('cat', cat),
                ('xgb :', xgb),
                ('lgb :', lgb)],
    n_jobs=1,
)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가
from sklearn.metrics import accuracy_score, mean_squared_error
# classifiers = [rf_model, dt_model, nc_model
# classifiers = [cat, xgb, lgb]
# for model in classifiers:
#     model.fit(x_train, y_train)
#     y_pred = model.predict(x_test)
#     score = accuracy_score(y_pred, y_test)
#     class_names = model.__class__.__name__
#     print(f'{class_names} 정확도: {score:.4f}')
    # nc_model, rf_model, dt_model, 
regressors = [cat, xgb, lgb]
for regressor in regressors:
    y_pred = regressor.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    class_names = regressor.__class__.__name__
    print(f'{class_names} MSE: {mse:.4f}')
    
result = model.score(x_test, y_test)
print('voting 결과 :', result)

