import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
import time 

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
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#kfold
n_splits = 5
random_state = 62
kfold = KFold(  #kFold=회귀모델 / StratifiedKFold = 분류모델
    n_splits=n_splits,
    random_state=random_state,
    shuffle=True
)

#2. 모델
rf_model = RandomForestRegressor()
model = BaggingRegressor(       
    rf_model,  
    n_estimators=100,
    n_jobs=-1,
    random_state=42
)

#3. 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time() - start_time

#3. 훈련, 평가
result = cross_val_score(
    model, 
    x, y,
    cv=kfold
)

print('r2 score :', result)