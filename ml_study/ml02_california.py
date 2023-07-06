from sklearn.svm import SVR, LinearSVR
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

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

#2. 모델
model = LinearSVR()
# model = SVR()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가
result = model.score(x_test, y_test)
print('R2 score :', result)  # 회귀모델의 model.score값은 r2 score

# LinearSVR() R2 score :  -0.5071457522413036
#SVR R2 score : -0.030050134514531868