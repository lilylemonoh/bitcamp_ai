from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR, SVC
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

#1. 데이터
#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target



x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.3,
    random_state=72,
    shuffle=True
)


# scaler
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


n_splits=7
random_state=1234
kfold = StratifiedKFold(
    n_splits=n_splits,
    shuffle=True,
    random_state=random_state
)

#2. 모델
# model = SVC()
# model = RandomForestClassifier()
model = DecisionTreeClassifier()

#3. 훈련
model.fit(x_train, y_train)


#4. 평가
score = cross_val_score(
    model, 
    x, y,
    cv=kfold
)

print('acc score :', score, 
      '\n cross_val_score:', round(np.mean(score), 4))


################################################feature importance는 시각화
print(model, ":", model.feature_importances_)

import matplotlib.pyplot as plt

n_features = datasets.data.shape[1]
plt.barh(range(n_features), model.feature_importances_,
         align='center')
plt.yticks(np.arange(n_features), datasets.feature_names)
plt.title('Cancer Feature Importances')
plt.ylabel('Feature')
plt.xlabel('Importances')
plt.ylim(-1, n_features)
plt.show()



#SVR
# acc score : [0.87234838 0.84043898 0.88140127 0.89890201 0.8210232  0.84422392
#  0.83340942]
#  cross_val_score: 0.856

# RandomForestClassifier()
# acc score : [0.97560976 0.93902439 0.97530864 0.96296296 0.9382716  0.98765432
#  0.96296296]
#  cross_val_score: 0.9631

