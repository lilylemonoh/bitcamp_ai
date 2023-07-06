from keras.preprocessing.text import Tokenizer

text1 = '나는 아침에 진짜 매우 매우 매우 매우 맛있는 밥을 \
    엄청 많이 많이 많이 먹어서 매우 배가 부르다'
text2 = '나는 인공지능이 정말 재미있다. 재밌어하는 내가 \
    너무 너무 너무 너무 너무 멋있다.'
    
token = Tokenizer()
token.fit_on_texts([text1, text2]) # fit_on 하면서 index 생성

print(token.word_index)
# {'매우': 1, '너무': 2, '많이': 3, '나는': 4, '아침에': 5, '진짜': 6, '맛있는': 7, 
#  '밥을': 8, '엄청': 9, '먹어서': 10, '배가': 11, '부르다': 12, '인공지능이': 13, 
#  '정말': 14, '재미있다': 15, '재밌어하는': 16, '내가': 17, '멋있다': 18} 

x = token.texts_to_sequences([text1, text2])
print(x)
# [[4, 5, 6, 1, 1, 1, 1, 7, 8, 9, 3, 3, 3, 10, 1, 11, 12], 
#  [4, 13, 14, 15, 16, 17, 2, 2, 2, 2, 2, 18]]
# print(x.shape)  # AttributeError: 'list' object has no attribute 'shape'
    
from keras.utils import to_categorical #to_categorical 하면 index수 + 1개가 만들어짐

x_new = x[0] + x[1] # list 2개를 1개로 합침
print(x_new) 
# x_new = to_categorical(x_new)
# print(x_new)
# print(x_new.shape) #(29, 19) #인덱스가 18까지인데 19로 만들어짐

################################ one hot encoding 수정 #############################
from sklearn.preprocessing import OneHotEncoder
import numpy as np

onehot_encoder = OneHotEncoder(
    categories='auto', sparse=False
)

x = np.array(x_new)
print(x.shape) #(29,) 단어개수
x = x.reshape(-1, 1) # 1차원을 2차원으로 만들어 줌
print(x.shape) #(29, 1)

onehot_encoder.fit(x)
x = onehot_encoder.transform(x)
print(x)
print(x.shape) #(29, 18)


