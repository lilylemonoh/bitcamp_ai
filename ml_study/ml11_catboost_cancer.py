from catboost import CatBoostClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
import numpy as np



#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target


# scaler
scaler = RobustScaler()
x = scaler.fit_transform(x)


#kfold
n_splits=7
random_state=1234
kfold = StratifiedKFold(
    n_splits=n_splits,
    random_state=random_state,
    shuffle=True
)

#2. 모델
model = CatBoostClassifier()

#3. 훈련, 평가
score = cross_val_score(
    model,
    x, y,
    cv = kfold
)
print('acc score :', score,
      '\n cross_val_score :', round(np.mean(score), 4))



# acc score : [0.97560976 0.96341463 0.97530864 0.96296296 0.96296296 0.97530864
#  0.96296296]

# xgb cross_val_score : 0.9613
# lgbm cross_val_score : 0.9666
# catboost  cross_val_score : 0.9684

# 분석 결과 catboost가 가장 성능이 좋다.