import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense






# 1. 데이터
#1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]]
y_data = [0,1,1,0]
# x_data = np.array(x_data)
# print(x_data.shape) #(4,2)

#2. 모델
model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))

# sklearn의 Perceptron 모델과 동일하다

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x_data,y_data,
          epochs=100,
          batch_size=16)

#4. 평가, 예측
#result = model.score(x_data, y_data)

loss,acc = model.evaluate(x_data, y_data)
print('loss :', loss)
print('acc:', acc)


y_predict = model.predict(x_data)
print(x_data, '의 예측결과', y_predict)

