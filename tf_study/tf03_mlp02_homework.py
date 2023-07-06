import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 2, 1, 1, 2, 1.1, 1.2, 1.4, 1.5, 1.6],
              [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

# 모델구성부터 평가예측까지 완성하시오
# 예측 [[10, 1.6, 1]]
print(x.shape)
print(y.shape)
# x : (3, 10) y : (10, 0)
x = x.transpose()
print(x.shape)
# x : (10, 3)

# 2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(20))
model.add(Dense(30))
model.add(Dense(40))
model.add(Dense(50))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=256)

# 4. 평가, 예측
loss = model.evaluate(x, y)
print('loss :', loss)

result = model.predict([[10, 1.6, 1]])
print('result :', result)  #20
