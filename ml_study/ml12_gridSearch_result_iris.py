import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
import time

#1. 데이터
datasets = load_iris()
x = datasets.data
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
kfold = StratifiedKFold(
    n_splits=n_splits,
    random_state=random_state,
    
    shuffle=True
)

param = [
    {'n_estimators' : [100, 500], 'max_depth':[6, 8, 10, 12], 'n_jobs' : [-1, 2, 4]},  
    {'max_depth' : [6, 8, 10, 12], 'min_samples_split' : [2, 3, 5, 10]},
    {'n_estimators' : [100, 200], 'min_samples_leaf' : [3, 5, 7, 10]},
    {'min_samples_split' : [2, 3, 5, 10], 'n_jobs' : [-1, 2, 4]}, 
    {'n_estimators' : [100, 200],'n_jobs' : [-1, 2, 4]}
]

#2. 모델
model = RandomForestClassifier(max_depth=10, n_jobs=4)
# model = GridSearchCV(
#     rf_model,   #모델
#     param,      #하이퍼 파라미터
#     cv=kfold, 
#     refit=True,  
#     verbose=1,
#     n_jobs=-1
# )

#3. 훈련
start_time = time.time()
model.fit(x, y)
end_time = time.time() - start_time

# print('최적의 파라미터 :', model.best_params_)
# print('최적의 매개변수 :', model.best_estimator_)
# print('best_score :', model.best_score_)
print('model_score :', model.score(x, y))
print('걸린 시간 :', end_time)


# 최적의 파라미터 : {'max_depth': 10, 'n_estimators': 100, 'n_jobs': 4}
# 최적의 매개변수 : RandomForestClassifier(max_depth=10, n_jobs=4)
# best_score : 0.9714285714285713
# model_score : 0.9333333333333333
# 걸린 시간 : 12.72077202796936