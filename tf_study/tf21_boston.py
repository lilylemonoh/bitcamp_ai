import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint
import time
import datetime

#1. 데이터
path = './data/boston/'
x_train = pd.read_csv(path + 'train-data.csv')
x_test = pd.read_csv(path + 'test-data.csv')
y_train = pd.read_csv(path + 'train-target.csv')
y_test = pd.read_csv(path + 'test-target.csv')

print(x_train.shape, x_test.shape) #(333, 12) (173, 12) # feature 11개 
print(y_train.shape, y_test.shape) #(333, 1) (173, 1)
print(x_train.columns) 
#Index(['zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax',
#       'ptratio', 'lstat'],
print(y_train.columns)
print(x_train.describe())
print(x_train.info())

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
 
# [실습] 코드를 완성하세요.
#2. 모델구성

model = Sequential()
model.add(Dense(64, input_dim=12))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')

# earlyStopping = EarlyStopping(
#     monitor='val_loss',
#     patience=50,
#     mode='min',
#     verbose=1,
#     restore_best_weights=True
# )

# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     verboss=1,
#     save_best_only=True,
#     filepath='./_mcp/tf21_boston.hdf5'
# )

model.fit(x_train, y_train, epochs=500, batch_size=32)         


#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss :', loss)

##### r2 score 결정계수 : 1에 가까울 수록 우수한 모델 #####
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print('r2 :',r2)

