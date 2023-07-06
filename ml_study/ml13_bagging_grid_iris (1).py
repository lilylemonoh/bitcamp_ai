# [실습] 
# 1. RandomForestClassifier 모델과 boosting 3대장 모델 
# 총 4개 모델을 적용하여 결과를 비교 해보자

from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split, \
                                    StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import MinMaxScaler
import time

# 1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    train_size=0.7,
    random_state=72,
    shuffle=True
)

# scaler 
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# kfold
n_splits = 5
random_state = 62
kfold = StratifiedKFold(
    n_splits=n_splits,
    random_state=random_state,
    shuffle=True
)

param = {
    'n_estimators' : [100],
    'random_state' : [42, 52, 62, 72],
    'max_features' : [3, 4, 7]
}

# 2. 모델
dt_model = DecisionTreeClassifier()
rf_model = RandomForestClassifier()
xgboost = XGBClassifier()
lgbm = LGBMClassifier()
catboost = CatBoostClassifier()
bagging = BaggingClassifier(
    # dt_model,
    # rf_model,
    # xgboost,
    lgbm,
    n_estimators=100,
    n_jobs=-1,
    random_state=random_state
)
model = GridSearchCV(
    bagging,
    param,
    cv=kfold,
    refit=True,
    n_jobs=-1,
    verbose=1
)
# 3. 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time() - start_time

# 4. 평가
result = model.score(x_test, y_test)
score = cross_val_score(
    model,
    x, y,
    cv=kfold
)
import numpy as np
print('acc score : ', result,
      '\n cross_val_score : ', round(np.mean(score), 4))

# 1. DecisionTreeClassifier
# acc score :  0.9555555555555556
#  cross_val_score :  0.9333

# 2. RandomForestClassifier
# acc score :  0.9333333333333333 
#  cross_val_score :  0.9333

# 3. xgboost
# acc score :  0.9555555555555556 
#  cross_val_score :  0.9333

# 4. lgbm
# acc score :  0.9333333333333333 
#  cross_val_score :  0.9533