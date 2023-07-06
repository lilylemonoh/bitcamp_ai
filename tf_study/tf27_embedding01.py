import numpy as np
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM

# 1. 데이터
docs = ['재미있어요', '재미없다', '돈 아깝다', '최고예요',
        '배우가 잘 생겼어요', '추천해요', '글쎄요', '감동이다',
        '최악', '후회된다', '보다 나왔다', '발연기예요',
        '꼭봐라', '세번봐라', '또보고싶다', '돈버렸다', 
        '다른 거 볼걸', 'n회차 관람', '다음편 나왔으면 좋겠다',
        '연기가 어색해요', '줄거리가 이상해요', '숙면했어요', 
        '망작이다', '차라리 집에서 잘걸', '즐거운 시간보냈어요']
# 긍정 1, 부정 0
labels = np.array([1, 0, 0, 1,
                  1, 1, 0, 1,
                  0, 0, 0, 0,
                  1, 1, 1, 0,
                  0, 1, 1, 0,
                  0, 0, 0, 0, 1])

#Tokenizer
token = Tokenizer()
token.fit_on_texts(docs) # index화

print(token.word_index)
# {'재미있어요': 1, '재미없다': 2, '돈': 3, '아깝다': 4, '최고예요': 5, 
#  '배우가': 6, '잘': 7, '생겼어요': 8, '추천해요': 9, '글쎄요': 10, 
#  '감동이다': 11, '최악': 12, '후회된다': 13, '보다': 14, '나왔 다': 15, 
#  '발연기예요': 16, '꼭봐라': 17, '세번봐라': 18, '또보고싶다': 19, '돈버렸다': 20, 
#  '다른': 21, '거': 22, '볼걸': 23, 'n회차': 24, '관람': 25, '다음편': 26, 
#  '나왔으면': 27, '좋겠다': 28, '연 기가': 29, '어색해요': 30, '줄거리가': 31, 
# '이상해요': 32, '숙면했어요': 33, '망작이다': 34, '차라리': 35, '집에서': 36, 
# '잘걸': 37, '즐거운': 38, '시간보냈어요': 39}

x = token.texts_to_sequences(docs)
print(x) # 3자리가 가장 길다 

# pad_sequences 
pad_x = pad_sequences(x, padding='pre', maxlen=3) #pre: 앞에서부터 0 넣겠다 maxlen:제일 긴 숫자
print(pad_x)
print(pad_x.shape) #(25, 3)

#word_size = input_dim의 개수 // 모델링하기 전에 단어의 개수 확인해야 함. 
word_size = len(token.word_index)
print('word_size :', word_size) # word_size : 39

# 모델 구성
model = Sequential()
model.add(Embedding(input_dim=40, # input_dim = word_size + 1
                    output_dim=16,  # output_dim = 노드의 수
                    input_length=3 # imput_length = 문장의 길이
                    )) 
model.add(LSTM(32)) # 순차적, 시간 순서가 영향을 미치는 것을 분석할때
# 문장은 시간의 순서가 중요하므로 LSTM 모델 사용
model.add(Dense(1, activation='sigmoid')) # 긍정과 부정의 이진분류

# 컴파일, 훈련
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy']
              )
model.fit(pad_x, labels, epochs=100,
          batch_size=32)

#4. 평가, 예측
loss, acc = model.evaluate(pad_x, labels)
print('loss :', loss)
print('acc :', acc)