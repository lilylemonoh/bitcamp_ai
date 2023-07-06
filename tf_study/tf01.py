#1. 데이터
import numpy as np
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])
#2. 모델구성
#버전마다 라이브러리 불러오는 위치가 다르다
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(3, input_dim=1))    # 입력층
model.add(Dense(4))                 # 히든레이어 1
# model.add(Dense(5))                 # 히든레이어 2
model.add(Dense(10))                 # 히든레이어 3
model.add(Dense(4))                 # 히든레이어 4
model.add(Dense(1))                 # 출력층

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x, y, epochs=100) # epochs는 몇 번 훈련할 건지

#4. 평가, 예측
loss = model.evaluate(x, y)
print('loss :', loss) #0.001536934170871973

result = model.predict([4])
print('4의 예측값 :', result) # [[3.914559]]