# 1. 데이터
import numpy as np

x = np.array([1, 2, 3, 5, 4])
y = np.array([1, 2, 3, 4, 5])

# 2. 모델구성
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(7))
model.add(Dense(3))
model.add(Dense(10))
model.add(Dense(4))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam') # mse에서 mae로 바꾸기
model.fit(x, y, epochs = 1000)

# 4. 평가, 예측
loss = model.evaluate(x, y)
print('loss : ', loss)
#mse loss : 0.3800027370452881
#mae lose : 0.40736064314842224

result = model.predict([6])
print('6의 예측값 :', result)
#mse 6의 예측값 : 5.6975207
#mae 6의 예측값 : 5.901524