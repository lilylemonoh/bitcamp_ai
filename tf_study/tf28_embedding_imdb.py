import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout
from keras_preprocessing.sequence import pad_sequences
from keras.datasets import imdb

#1. 데이터
(x_train, y_train),(x_test, y_test) = imdb.load_data(num_words=10000)
# x데이터는 자연어, y 데이터는 라벨 
# print(x_train)
print(x_train.shape, y_train.shape) #(25000,) #(25000,)
print(np.unique(y_train, return_counts=True))
# (array([0, 1], dtype=int64), array([12500, 12500], dtype=int64))
# 0은 부정 12500개, 1은 긍정 12500개

# 최대 길이와 평균 길이
print('x_train 리뷰의 최대 길이 : ', max(len(i) for i in x_train))
print('x_test 리뷰의 최대 길이 : ', max(len(i) for i in x_test))
# 리뷰의 최대 길이 :  2494(train) / 2315(test)
print('x_train 리뷰의 평균 길이 : ', sum(map(len, x_train)) / len(x_train))
# 리뷰의 평균 길이 :  238.71364

#pad_sequences
x_train = pad_sequences(x_train,
                        padding='pre',
                        maxlen=2494,
                        truncating='pre')
x_text = pad_sequences(x_test,
                       padding='pre',
                       maxlen=2494,
                       truncating='pre')
print(x_train.shape, x_test.shape) #(25000, 2494) (25000,)

# 2. 모델구성
model=Sequential()
model.add(Embedding(input_dim=10001,
                    output_dim=128,
                    input_length=2494
                    ))
model.add(LSTM(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(x_train, y_train, 
          epochs=100,
          batch_size=32,
          validation_split=0.2)

# 4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss :', loss)
print('acc :', acc)