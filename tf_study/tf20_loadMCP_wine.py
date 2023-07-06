from keras.models import Sequential, load_model
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from keras.utils import to_categorical
import time
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, MaxAbsScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) # (178, 13) (178,) # feature 13개
print(datasets.feature_names)
print(datasets.DESCR)
    # - class:
    #         - class_0
    #         - class_1
    #         - class_2
# one-hot encoding
y=to_categorical(y) # 클래스가 3개라서 3개로 분류하는 것
print(y.shape) #(178, 3) 원핫인코딩 후 (178,)에서 변경됨.

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.6,
    random_state=727,
    shuffle=True
)

print(x_train.shape, x_test.shape) # (124, 13) (54, 13)
print(y_train.shape, y_test.shape) # (124, 3) (54, 3)

#1 StandardScaler 적용
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# #2. 모델구성
# model = Sequential()
# model.add(Dense(32, input_dim=13))
# model.add(Dense(128))
# model.add(Dense(128))
# model.add(Dense(128))
# model.add(Dense(3, activation='softmax')) #0~1 사이로 바꿔줌 

# #3. 컴파일 훈련
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['mse', 'accuracy'])


# earlyStopping = EarlyStopping(
#     monitor='val_loss',
#     patience=50,
#     mode='min',
#     verbose=1,
#     restore_best_weights=True
# )

# #file명 생성
# import datetime
# date = datetime.datetime.now()
# date = date.strftime("%m%d_%H%M")

# filepath = './_mcp/'
# filename= '{epoch:04d}-{val_loss:.4f}.hdf5'
# mcp = ModelCheckpoint(
#     monitor='val_loss',
#     mode='auto',
#     verbose=1,
#     save_best_only=True,
#     # filepath='./_mcp/tf20_wine.hdf5'
#     filepath="".join([filepath, 'tf20_wine', date, '_', filename])
# )


# start_time=time.time()
# model.fit(x_train, y_train, 
#           validation_split=0.2,
#           callbacks=[earlyStopping, mcp],
#           epochs=500, batch_size=32)
# end_time=time.time() - start_time

 
 # load model
model = load_model('./_mcp/20_0628_1513_0020-0.2484.hdf5')
 
 
 
#4. 평가, 예측
loss, mse, accuracy = model.evaluate(x_test, y_test)
print('loss :', loss)
print('mse :', mse)
print('accuracy :', accuracy)


# loss : 0.0530683733522892
# mse : 0.010717528872191906
# accuracy : 0.9814814925193787
# 걸린 시간 : 4.172143220901489

#===================
# Epoch 61: early stopping
# 1/3 [=========>....................] - ETA: 0s - loss: 0.1558 - mse: 0.0323/3 [==============================] - 0s 1ms/step - loss: 0.0826 - mse: 0.0175 - accuracy: 0.9583
# loss : 0.08257784694433212
# mse : 0.017519226297736168
# accuracy : 0.9583333134651184
# 걸린 시간 : 2.469545602798462
