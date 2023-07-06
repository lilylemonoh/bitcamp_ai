import numpy as np
from keras.models import Sequential
from keras.layers import Dense 
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import time

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

print(x.shape) #(150, 4)
print(y.shape) #(150,)
print(datasets.feature_names) #['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
print(datasets.DESCR)
#        - class: (DESCR 로 확인해야 함)
#                - Iris-Setosa
#                - Iris-Versicolour
#                - Iris-Virginica

########## 원핫인코딩 one-hot encoding ########
from keras.utils import to_categorical
y = to_categorical(y)
print(y.shape) #(150, 3) (원래 : #(150,))


x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=72,
    shuffle=True
)

print(x_train.shape, x_test.shape) #(105, 4) (45, 4)  //// 4 -> input 숫자
print(y_train.shape, y_test.shape) #(105,) (45,) ---> (105, 3) (45, 3) (원핫인코딩 후 변경됨)  //// 3 -> output숫자

#2. 모델구성
model = Sequential()
model.add(Dense(32, input_dim=4))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['mse', 'accuracy']) 
# 회귀분석은 mse, r2 score
# 분류분석은 mse, accuracy score 들어간다.
start_time = time.time()
model.fit(x_train, y_train,
          epochs=500,
          batch_size=32)
end_time = time.time() - start_time
print('걸린시간 : ', end_time)

#4. 평가, 예측
loss, mse, accuracy = model.evaluate(x_test, y_test)
print('loss: ', loss)
print('mse :', mse)
print('accuracy :', accuracy)

# loss:  0.021603668108582497
# mse : 0.0023043793626129627
# accuracy : 1.0



