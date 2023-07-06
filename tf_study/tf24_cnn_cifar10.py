from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Dropout, Flatten, Conv2D, BatchNormalization
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import time

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)

#시각화
# plt.imshow(x_train[12])
# plt.show()

#Scaling
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train/255
x_test = x_test/255

#2. 모델 구성
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3, 3),
                 padding='same',
                 activation='relu',
                 input_shape=(32, 32, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.3))
model.add(Conv2D(128, (3,3),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation= 'softmax'))

# 3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

earlyStopping = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=50,
    verbose=1,
    restore_best_weights=True
)

start_time = time.time()
model.fit(
    x_train, y_train, validation_split=0.2, 
    callbacks=[earlyStopping],
    epochs=100, batch_size=32
)
end_time = time.time() - start_time

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss :', loss)
print('acc :', acc)
print('걸린 시간 :', end_time)

# Epoch 22: early stopping
# 313/313 [==============================] - 2s 6ms/step - loss: 0.9265 - accuracy: 0.6859
# loss : 0.9264764785766602
# acc : 0.6858999729156494
# 걸린 시간 : 373.66124081611633

# loss : 0.8487501740455627
# acc : 0.7050999999046326
# 걸린 시간 : 544.0778830051422

# loss : 0.8061368465423584
# acc : 0.7293000221252441
# 걸린 시간 : 934.2053225040436
 
