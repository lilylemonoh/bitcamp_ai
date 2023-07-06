from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from keras.callbacks import EarlyStopping

# 1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

print(x.shape)
print(y.shape)
print(datasets.feature_names)
print(datasets.DESCR)

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=1234,
    shuffle=True
)

print(x_train.shape, x_test.shape) #(398, 30) (171, 30)
print(y_train.shape, y_test.shape) # (398,) (171,)

#Scaler 적용
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성
model = Sequential()
model.add(Dense(68, input_dim=30))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(128))
model.add(Dense(1, activation='sigmoid')) 
# 이진분류는 마지막 아웃풋 레이어에 무조건 sigmoid 함수 사용



# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['mse', 'accuracy'])

earlyStopping = EarlyStopping(
    monitor = 'val_loss',
    patience=50,
    mode='min',
    verbose=1,
    restore_best_weights=True
)

model.fit(x_train, y_train, 
          validation_split = 0.2,
          callbacks = [earlyStopping],
          epochs=5000, batch_size=128)

# model 모델 저장
model.save('./_save/tf18_cancer.h5')

# 4. 평가, 예측
loss, mse, accuracy = model.evaluate(x_test, y_test)
print('loss :', loss) #loss : 1.7872555255889893
print('mse : ', mse) #mse :  0.2297373265028
print('accuracy : ', accuracy) #accuracy :  0.7543859481811523

#==========================================
# 1. StandardScaler
# loss : 0.3610512614250183
# mse :  0.02716873213648796
# accuracy :  0.9707602262496948

#2. MinMaxScaler
# loss : 0.3351745903491974
# mse :  0.052698347717523575
# accuracy :  0.9415204524993896

#3. MaxAbsScaler
# loss : 0.16473504900932312
# mse :  0.03488897904753685
# accuracy :  0.9590643048286438

#4. RobustScaler
# loss : 0.20332148671150208
# mse :  0.029410038143396378
# accuracy :  0.9707602262496948


# Epoch 66: early stopping
# 4/4 [==============================] - 0s 1ms/step - loss: 0.1639 - mse: 0.0353 - accuracy: 0.9561
# loss : 0.16389784216880798
# mse :  0.035275280475616455
# accuracy :  0.9561403393745422

#=====
# Epoch 65: early stopping
# 1/4 [======>.......................] - ETA: 0s - loss: 0.0039 - mse: 8.1883e-05 - a4/4 [==============================] - 0s 1ms/step - loss: 0.1540 - mse: 0.0356 - accuracy: 0.9474
# loss : 0.15395145118236542
# mse :  0.03564348816871643
# accuracy :  0.9473684430122375