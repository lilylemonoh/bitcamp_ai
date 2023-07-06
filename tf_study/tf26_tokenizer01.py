from keras.preprocessing.text import Tokenizer

text = '나는 아침에 진짜 매우 매우 매우 매우 맛있는 밥을 \
    엄청 많이 많이 많이 먹어서 매우 배가 부르다'
    
token = Tokenizer()
token.fit_on_texts([text])
# 단어의 빈도에 따라 숫자 부여

print(token.word_index)

x = token.texts_to_sequences([text])
print(x)


from keras.utils import to_categorical 
x = to_categorical(x) # 원핫인코딩
print(x)
print(x.shape) #(1, 17, 12) 3차원