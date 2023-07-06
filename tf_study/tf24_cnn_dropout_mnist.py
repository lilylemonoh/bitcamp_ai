import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.callbacks import EarlyStopping
import time

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)

print(x_train[0])
print(y_train[0])

#시각화
# import matplotlib.pyplot as plt
# plt.imshow(x_train[0], 'gray')
# plt.show()

# reshape // 채널 공간 1을 만들어주는 것.  y는 정답지라서 reshape할 필요가 없다.
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
print(x_train.shape, x_test.shape)

# 2. 모델구성
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(4,4), #kernal_size는 이미지를 자르는 규격을 의미
                 padding='same',
                 input_shape=(28, 28, 1))) #(rows, rows, channels)의 형태 channer 1은 블랙, 3은 컬러
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.2))
model.add(Conv2D(16, (2, 2),
                 padding='valid',
                 activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

earlyStopping = EarlyStopping(
    monitor = 'val_loss',
    mode='min',
    verbose=1,
    patience=5,
    restore_best_weights=True
)


start_time = time.time()
model.fit(x_train, y_train, 
          validation_split=0.2,
          callbacks=[earlyStopping],
          epochs=50, batch_size=32, verbose=1)
end_time=time.time() - start_time

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)

print('loss :', loss)
print('acc :', acc)
print('걸린 시간:', end_time)


# Epoch 9: early stopping
# 313/313 [==============================] - 2s 5ms/step - loss: 0.0817 - accuracy: 0.9778
# loss : 0.08173239231109619
# acc : 0.9778000116348267
# 걸린 시간: 142.80307960510254