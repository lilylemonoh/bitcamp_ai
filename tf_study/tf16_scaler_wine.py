from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from keras.utils import to_categorical
import time
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, MaxAbsScaler

# 1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (178, 13) (178,) # feature 13개
print(datasets.feature_names)
print(datasets.DESCR)
    # - class:
    #         - class_0
    #         - class_1
    #         - class_2
# one-hot encoding
y=to_categorical(y) # 클래스가 3개라서 3개로 분류하는 것
print(y.shape) #(178, 3) 원핫인코딩 후 (178,)에서 변경됨.

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=727,
    shuffle=True
)

print(x_train.shape, x_test.shape) # (124, 13) (54, 13)
print(y_train.shape, y_test.shape) # (124, 3) (54, 3)

#1 StandardScaler 적용
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
model = Sequential()
model.add(Dense(32, input_dim=13))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(3, activation='softmax')) #0~1 사이로 바꿔줌 

#3. 컴파일 훈련
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['mse', 'accuracy'])

start_time=time.time()
model.fit(x_train, y_train, epochs=500, batch_size=32)
end_time=time.time() - start_time

 
#4. 평가, 예측
loss, mse, accuracy = model.evaluate(x_test, y_test)
print('loss :', loss)
print('mse :', mse)
print('accuracy :', accuracy)
print('걸린 시간 :', end_time)

# loss : 0.0530683733522892
# mse : 0.010717528872191906
# accuracy : 0.9814814925193787
# 걸린 시간 : 4.172143220901489

#===================