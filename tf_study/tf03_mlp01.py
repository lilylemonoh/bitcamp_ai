import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 1. 데이터
x = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 2.1, 3.1, 4.1, 5.1, 6, 7, 8.1, 9.2, 10.5]])
# 2행 10열. 
# x는 환율, 주가 등이 될 수 있다. 지금은 주가라고 가정함.
# 11에 영향을 미친 건 1,1 / 12에 영향을 미친 건 2, 2.1 ... 이다.
y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
# 10행

print(x.shape) #(2, 10)
print(y.shape) #(10,) 
# x와 y의 행이 같아야 하는 것이 기본이다. 
# 그래야 데이터를 분석할 수 있다.

x = x.transpose()   # 동일한 코드 x = x.T
print(x.shape) #(10, 2)로 바뀜.
# 2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=2))
model.add(Dense(55))
model.add(Dense(50))
model.add(Dense(30))
model.add(Dense(3))
model.add(Dense(10))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=1000, batch_size=16)

# 4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)

result = model.predict([[10, 10.5]]) #20이 나와야..
print('10과 10.5의 예측값 :', result)

