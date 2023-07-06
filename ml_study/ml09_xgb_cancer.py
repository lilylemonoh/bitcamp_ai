from xgboost import XGBClassifier
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
model = XGBClassifier()

#3. 훈련, 평가
score = cross_val_score(
    model,
    x, y,
    cv = kfold
)
print('acc score :', score,
      '\n cross_val_score :', round(np.mean(score), 4))


# acc score : [0.97560976 0.95121951 0.97530864 0.95061728 0.96296296 0.96296296
#  0.95061728]

# xgb cross_val_score : 0.9613
