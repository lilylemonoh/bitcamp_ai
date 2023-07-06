from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, LinearSVC
import numpy as np
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

#1. 데이터
datasets = load_iris()
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
# nc_model = KNeighborsClassifier(n_neighbors=7)
# rf_model = RandomForestClassifier()
# dt_model = DecisionTreeClassifier()

cat= CatBoostClassifier()
xgb = XGBClassifier()
lgb = LGBMClassifier()


model = VotingClassifier(
    # estimators=[('nc_model', nc_model),
    #             ('rf_model :', rf_model),
    #             ('dt_model :', dt_model)],
    estimators = [('cat', cat),
                ('xgb :', xgb),
                ('lgb :', lgb)],
    voting='soft',
    n_jobs=1
)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가
from sklearn.metrics import accuracy_score
# classifiers = [rf_model, dt_model, nc_model
classifiers = [cat, xgb, lgb]
for model in classifiers:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = accuracy_score(y_pred, y_test)
    class_names = model.__class__.__name__
    print(f'{class_names} 정확도: {score:.4f}')
    
result = model.score(x_test, y_test)
print('voting 결과 :', result)

#{0} 정확도 : {1; .4f}
# {0} 정확도 : {1; .4f}
# {0} 정확도 : {1; .4f}
# voting 결과 : 0.9333333333333333

# CatBoostClassifier 정확도: 0.9556
# XGBClassifier 정확도: 0.9333
# LGBMClassifier 정확도: 0.9556
# voting 결과 : 0.9555555555555556