from sklearn.svm import SVC, LinearSVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

datasets = load_iris()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.3,
    random_state=1234,
    shuffle=True
)

print(x_train.shape, y_train.shape) #(105, 4) (105,)
print(x_test.shape, y_test.shape) # (45, 4) (45,)


# Scaler (정규화) 적용
from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler() # 표준화 scaler
# scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



#2. 모델
# model = LinearSVC()
model = SVC()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가
result = model.score(x_test, y_test)
print('accuracy', result) # 분류모델은 score = accuracy
#linearSVC result :0.9555555555555556
#SVC result: 0.9777777777777777


# StandardScaler 적용 후 accuracy 0.9555555555555556
# MinMaxScaler() 적용 후 accuracy 0.9555555555555556
# MaxAbsScaler() 적용 후 accuracy 0.9555555555555556
# RobustScaler()  적용 후 accuracy 0.9555555555555556