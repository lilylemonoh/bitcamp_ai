import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array(range(1, 20))
y = np.array(range(1, 20))


# 순서 중요
x_train, x_test, y_train, y_test = train_test_split(
    x, y,   # 데이터
    test_size=0.3, # test set 30%
    train_size=0.7, # train set 70%
    random_state=2023, #데이터를 난수값에 의해 추출한다는 의미
    shuffle=True # 순차적인 데이터는 shuffle을 false로(예 : 날씨)
)

print(x_train, y_train)
print(x_test, y_test)

#2. 모델구성

model = Sequential()
model.add(Dense(13, input_dim=1))
model.add(Dense(100))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=32)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

result = model.predict([21])
print('result :', result)