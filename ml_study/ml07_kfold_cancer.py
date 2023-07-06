from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR
import numpy as np
from sklearn.ensemble import RandomForestClassifier

#1. 데이터
#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

# scaler // test, train 나누지 않음
scaler = RobustScaler()
x = scaler.fit_transform(x)


n_splits=7
random_state=1234
kfold = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=random_state
)

#2. 모델
# model = SVR()
model = RandomForestClassifier()

#3. 훈련, 평가
score = cross_val_score(
    model, 
    x, y,
    cv=kfold
)

print('acc score :', score, 
      '\n cross_val_score:', round(np.mean(score), 4))

#SVR
# acc score : [0.87234838 0.84043898 0.88140127 0.89890201 0.8210232  0.84422392
#  0.83340942]
#  cross_val_score: 0.856

# RandomForestClassifier()
# acc score : [0.97560976 0.93902439 0.97530864 0.96296296 0.9382716  0.98765432
#  0.96296296]
#  cross_val_score: 0.9631

