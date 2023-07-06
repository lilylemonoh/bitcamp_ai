from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
import numpy as np

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
print(x.shape)


#drop feature
x = np.delete(x, [3,4], axis=1)
print(x.shape) #(20640, 6)



x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=72,
    shuffle=True
)

# scaler 
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits=7
random_state=72
kfold = KFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=random_state
)

#2. 모델
model = DecisionTreeRegressor()

#3. 훈련
model.fit(x_train, y_train)


#4. 평가
score = cross_val_score(
    model, 
    x, y,
    cv=kfold
)

print('r2 score :', score, '\n cross_val_score:',
      round(np.mean(score), 4))

# r2 score : [0.73330084 0.75016781 0.73163992 0.73693374 0.74697454] 
#  cross_val_score: 0.7398


################################################feature importance는 시각화
# print(model, ":", model.feature_importances_)

# import matplotlib.pyplot as plt

# n_features = datasets.data.shape[1]
# plt.barh(range(n_features), model.feature_importances_,
#          align='center')
# plt.yticks(np.arange(n_features), datasets.feature_names)
# plt.title('california Feature Importances')
# plt.ylabel('Feature')
# plt.xlabel('Importances')
# plt.ylim(-1, n_features)
# plt.show()
