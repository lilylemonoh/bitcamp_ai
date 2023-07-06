import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
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

param = [
    {'n_estimators' : [100, 500], 'max_depth':[3, 6, 8, 10, 12], 'n_jobs' : [-1, 2, 4]},  
    {'max_depth' : [6, 8, 10, 12], 'min_samples_split' : [2, 3, 5, 10]},
    {'n_estimators' : [100, 200], 'min_samples_leaf' : [3, 5, 7, 10]},
    {'min_samples_split' : [2, 3, 5, 10], 'n_jobs' : [-1, 2, 4]}, 
    {'n_estimators' : [100, 200],'n_jobs' : [-1, 2, 4]}
]


#2. 모델
rf_model = RandomForestRegressor()
dt_model = DecisionTreeRegressor()
model = GridSearchCV(       #RandomizedSearch
    rf_model,   #모델
    param,      #하이퍼 파라미터
    cv=kfold, 
    refit=True,  
    verbose=1,
    n_jobs=-1
)

#3. 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time() - start_time

print('최적의 파라미터 :', model.best_params_)
print('최적의 매개변수 :', model.best_estimator_)
print('best_score :', model.best_score_)
print('model_score :', model.score(x_test, y_test))
print('걸린 시간 :', end_time)

# 최적의 파라미터 : {'min_samples_split': 2, 'n_jobs': -1}
# 최적의 매개변수 : RandomForestRegressor(n_jobs=-1)
# best_score : 0.8017239448393333
# model_score : 0.8108202867820838
# 걸린 시간 : 600.5586471557617