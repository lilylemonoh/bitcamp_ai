import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import time
import datetime

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1./255,         # 스케일 조정
    horizontal_flip=True,   # 수평으로 반전
    width_shift_range=0.1,  # 수평 이동 범위
    height_shift_range=0.1, # 수직 이동 범위
    rotation_range=5,       # 회전 범위
    zoom_range=1.2,         # 확대 범위
    shear_range=0.5,        # 기울이기 범위
    fill_mode='nearest',
    validation_split=0.2 # 전체 이미지의 0.2퍼센트가 validation으로 넘어간다.
        # 트레인, 테스트 데이터가 분리되어 있지 않을 때 필요하다.
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

xy_train = train_datagen.flow_from_directory(
    './data/horse-or-human/',
    target_size=(150, 150),
    batch_size=822,
    class_mode='binary',
    color_mode='rgb',
    shuffle=True,
    subset='training'
)

print(xy_train[0][0].shape) #x_train (822, 150, 150, 3)
print(xy_train[0][1].shape) #y_train (822,)

xy_test = train_datagen.flow_from_directory(
    './data/horse-or-human/',
    target_size=(150, 150),
    batch_size=128,
    class_mode='binary',
    color_mode='rgb',
    subset='validation'
)

print(xy_test[0][0].shape) # x_test (128, 150, 150, 3)
print(xy_test[0][1].shape) # y_test (128,)

#[실습] 코드를 완성하시오
# 2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (2, 2), 
                 input_shape = (150, 150, 3),
                 activation='relu',
                 padding='same'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.2))
model.add(Conv2D(32, (2, 2), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])

earlyStopping =EarlyStopping(
    monitor='val_loss',
    mode='min',
    verbose=1,
    patience=5,
    restore_best_weights=True
)

date = datetime.datetime.now()
date = date.strftime("%m%d_%h%m")
filepath = './_mcp/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
mcp = ModelCheckpoint(
    monitor='val_loss',
    mod='auto',
    save_best_only=True,
    verbose=1,
    filepath="".join([filepath, 'tf25_horse_human', date, '_', filename])
)


start_time = time.time()
model.fit(xy_train[0][0], xy_train[0][1],
          validation_split=0.2,
          epochs=30, batch_size=128,
          verbose=1)
end_time=time.time() - start_time

#4. 평가, 예측
loss, acc = model.evaluate(xy_test[0][0], xy_test[0][1])

print('loss :', loss)
print('acc :', acc)
print('걸린 시간:', end_time)

# loss : 3.5680535308275524e-23
# acc : 1.0
# 걸린 시간: 282.2736234664917