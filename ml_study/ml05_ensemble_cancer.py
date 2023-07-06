from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

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

# Scaler (정규화) 적용
from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler() # 표준화 scaler
# scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델 
model = RandomForestClassifier()


#3. 훈련
model.fit(x_train, y_train)

#4. 평가
result = model.score(x_test, y_test)
print('result', result) # 분류모델은 score = accuracy
#linearSVC result : 0.8771929824561403
# svc : result 0.8947368421052632


# StandardScaler 적용 후 result 0.9415204678362573
# MinMaxScaler() 적용 후 result 0.9415204678362573
# MaxAbsScaler() 적용 후 result 0.9298245614035088
# RobustScaler()  적용 후 result 0.9298245614035088

# RobustScaler, DecisiontreeClassifier 적용 후 result 0.935672514619883

# RobustScaler,RandomForestClassifier() 적용 후 result 0.935672514619883