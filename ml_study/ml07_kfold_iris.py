from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC, LinearSVC
import numpy as np
from sklearn.ensemble import RandomForestClassifier


#1. 데이터
datasets = load_iris()
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
# model = SVC()
model = RandomForestClassifier()

#3. 훈련, 평가
score = cross_val_score(
    model, 
    x, y,
    cv=kfold
)

print('acc score :', score, 
      '\n cross_val_score:', round(np.mean(score), 4))


#SVC
# acc score : [1.         1.         0.95454545 0.9047619  0.95238095 0.9047619
#  0.95238095]
#  cross_val_score: 0.9527

#RandomForestClassifier()
# acc score : [1.         0.95454545 0.95454545 0.95238095 1.         0.9047619
#  0.95238095]
#  cross_val_score: 0.9598

