from xgboost import XGBRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import numpy as np

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

# scaler 
scaler = StandardScaler()
x = scaler.fit_transform(x)

#kfold
n_splits = 7
random_state = 72
kfold = KFold(
    n_splits=n_splits,
    random_state=random_state,
    shuffle=True
)


#2. 모델
model = XGBRegressor()


#3. 훈련, 평가
score = cross_val_score(
    model,
    x, y,
    cv = kfold
)
print('acc score :', score,
      '\n cross_val_score :', round(np.mean(score), 4))


# acc score : [0.83555214 0.8399611  0.82564844 0.84354762 0.8348756  0.8453127
#  0.83203637] 

# xgb  cross_val_score : 0.8367