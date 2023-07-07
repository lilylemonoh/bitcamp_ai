import numpy as np
from keras.models import Sequential 
from keras.layers import Dense, SimpleRNN
import time

#1. 데이터
datasets = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# 시계열 데이터이므로 현재 수업차시에서는 임의로 a개씩 자름
# y = ?
x = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9]]) 
y = np.array([4, 5, 6, 7, 8, 9, 10]) #(1,2,3 다음에 4 나와야 하고, ...)

print(x.shape) #(7, 3)
print(y.shape) #(7,)
# x의  shape = (행, 열, timesteps!!!) 
x = x.reshape(7, 3, 1)
print(x.shape)

#2. 모델구성
model = Sequential()
model.add(SimpleRNN(128, input_shape=(3,1)))
model.add(Dense(128, activation='relu')) # RNN 계열(lstn 등)은 3차원을 받아서 2차원으로 보낸다. 
model.add(Dense(64, activation='relu'))  # 즉 Flatten이 필요없다. -> 사용하기 간편함.
model.add(Dense(64, activation='relu')) 
model.add(Dense(64, activation='relu')) 
model.add(Dense(64, activation='relu')) 
model.add(Dense(32, activation='relu')) 
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam') 

start_time = time.time()
model.fit(x, y, epochs=700, batch_size=16)
end_time = time.time() - start_time


#4. 평가, 예측
loss = model.evaluate(x, y)
y_pred = np.array([8, 9, 10]).reshape(1, 3, 1)

result = model.predict(y_pred)
print('loss : ', loss)
print('[8, 9, 10]의  결과  : ', result)
print('걸린 시간 :', end_time)

# loss :  7.042143579383264e-07
# [8, 9, 10]의  결과  :  [[10.8829365]]
# 걸린 시간 : 3.342613458633423