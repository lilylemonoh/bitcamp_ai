from sklearn.svm import SVR, LinearSVR
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.tree import DecisionTreeRegressor

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.3,
    random_state=1234,
    shuffle=True
)

print(x_train.shape, y_train.shape) 
print(x_test.shape, y_test.shape)

# Scaler (정규화) 적용
from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler() # 표준화 scaler
scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)




# 2. 모델 
model = DecisionTreeRegressor()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가
result = model.score(x_test, y_test)
print('R2 score :', result)  # 회귀모델의 model.score값은 r2 score

# LinearSVR() R2 score :  -0.5071457522413036
#SVR R2 score : -0.030050134514531868
# StandardScaler 적용 후 R2 score : 0.3328914917050597
# MinMaxScaler() 적용 후 R2 score : 0.5818150463925045
# MaxAbsScaler() 적용 후 R2 score : 0.5372204225696585
# RobustScaler()  적용 후 R2 score : -0.8103372687435217

#MinMaxScaler(), DecisionTreeRegressor()  적용 후 R2 score : 0.6049773798438058