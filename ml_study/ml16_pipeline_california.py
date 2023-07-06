from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler #스케일러
from sklearn.ensemble import RandomForestRegressor #모델
from sklearn.pipeline import make_pipeline

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size= 0.7,
    random_state= 72,
    shuffle=True
)

# 2. 모델(파이프라인)
model = make_pipeline(
    StandardScaler(),
    RandomForestRegressor()
)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가
result = model.score(x_test, y_test)

print("r2 score : ", result)
