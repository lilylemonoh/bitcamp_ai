# [실습]
# 1. r2 score를 음수가 아닌 0.5 이하로 만드세요
# 2. 데이터는 건드리지 마세요
# 3. 레이어는 인풋, 아웃풋 포함 7개 이상 만드세요(히든레이어 5개 이상)
# 4. batch_size=1이어야 함
# 5. 히든레이어의 노드(뉴런) 개수는 10개 이상 100개 이하로 하세요
# 6. train_size=0.7로 하세요
# 7. epochs= 100 이상으로 하세요
# [실습시작]

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

#1. 데이터(tf04_train_test_split01.py 데이터 사용)
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]) # 14까지 트레인(70퍼센트), 이후 테스트(30퍼센트)
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])


x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    train_size=0.7,
    random_state=0,
    shuffle=False
)

print(x_train.shape)

#2. 모델 구성
model = Sequential()
model.add(Dense(100, input_dim=1))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 :', r2)