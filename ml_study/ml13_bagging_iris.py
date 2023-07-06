import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
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

# scaler 
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

#2. 모델
dt_model = DecisionTreeClassifier()
model = BaggingClassifier(
    dt_model, 
    n_estimators=100,
    n_jobs=-1,
    random_state=random_state
)

#3. 훈련
start_time = time.time()
model.fit(x_train, y_train)
end_time = time.time() - start_time

#4. 평가
result = model.score(x_test, y_test)
score = cross_val_score(
    model, 
    x, y,
    cv=kfold
)

print('acc score : ', result,
      '\n cross_val_score :', round(np.mean(score), 4))

#acc score :  0.9333333333333333 
#  cross_val_score : 0.9333