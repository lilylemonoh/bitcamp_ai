from sklearn.svm import SVC, LinearSVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

#[실습] SVC와 LinearSVC 모델을 적용하여 코드를 완성하시오.
#load 데이터 - x,y에 담기 - train test split

# 1. 데이터
datasets = load_breast_cancer()
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
# model = LinearSVC()
model = SVC()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가
result = model.score(x_test, y_test)
print('result', result) # 분류모델은 score = accuracy
#linearSVC result : 0.8771929824561403
# svc : result 0.8947368421052632