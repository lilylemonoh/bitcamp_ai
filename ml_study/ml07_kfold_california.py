from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import numpy as np

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

# 원래는 이렇게 쓴다.
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y,
#     train_size=0.7,
#     random_state=1234,
#     shuffle=True
# )

# scaler // test, train 나누지 않음
scaler = StandardScaler()
x = scaler.fit_transform(x)

kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=1234
)

#2. 모델
model = SVR()

#3. 훈련, 평가
score = cross_val_score(
    model, 
    x, y,
    cv=kfold
)

print('r2 score :', score, '\n cross_val_score:',
      round(np.mean(score), 4))

# r2 score : [0.73330084 0.75016781 0.73163992 0.73693374 0.74697454] 
#  cross_val_score: 0.7398


