import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# 1. 데이터
path = './data/credit_card_prediction/'
datasets = pd.read_csv(path + 'train.csv')

print(datasets.columns)

# Index(['ID', 'Gender', 'Age', 'Region_Code', 'Occupation', 'Channel_Code',
#        'Vintage', 'Credit_Product', 'Avg_Account_Balance', 'Is_Active',
#        'Is_Lead'], //11개 
#       dtype='object')

print('******Labeling 전 데이터*****')
print(datasets.head(8))

# ******Labeling 전 데이터*****
#          ID  Gender  Age Region_Code     Occupation Channel_Code  Vintage Credit_Product  Avg_Account_Balance Is_Active  Is_Lead
# 0  NNVBBKZB  Female   73       RG268          Other           X3       43             No              1045696        No        0
# 1  IDD62UNG  Female   30       RG277       Salaried           X1       32             No               581988        No        0
# 2  HD3DSEMC  Female   56       RG268  Self_Employed           X3       26             No              1484315       Yes        0
# 3  BF3NC7KV    Male   34       RG270       Salaried           X1       19             No               470454        No        0
# 4  TEASRWXV  Female   30       RG282       Salaried           X1       33             No               886787        No        0
# 5  ACUTYTWS    Male   56       RG261  Self_Employed           X1       32             No               544163       Yes        0
# 6  ETQCZFEJ    Male   62       RG282          Other           X3       20            NaN              1056750       Yes        1
# 7  JJNJUQMQ  Female   48       RG265  Self_Employed           X3       13             No               444724       Yes        0


# 문자를 숫자로 변경 (LabelEncoder)
df = pd.DataFrame(datasets)
# Pandas 라이브러리를 사용하여 데이터셋을 기반으로 데이터프레임(DataFrame) 객체를 생성
#dataset과 형태 동일

ob_col = list(df.dtypes[df.dtypes=='object'].index) # object 컬럼 리스트
#df에서 'object'형 객체를 찾아 그 열 이름을 리스트 형태로 저장
#print(ob_col)
#['ID', 'Gender', 'Region_Code', 'Occupation', 'Channel_Code', 'Credit_Product', 'Is_Active']

# NaN 처리
df['Credit_Product'].fillna('Unknown', inplace=True)
#df의 Credit_Product 열에서 결측치를 Unknown으로 대체하는 코드
#fillna()는 Pandas의 메서드로 데이터프레임에서 결측치를 대체하는 역할 
# inplace=True는대체된 결과를 원본 데이터 프레임에 반영하는 인자
# 대체된 결과가 원본 데이터프레임에 저장되며 새로운 데이터 프레임을 반환하지 않음
#-> 원본 데이터 프레임이 직접 수정됨


for col in ob_col:
    df[col] = LabelEncoder().fit_transform(df[col].values)
#df에서 객체형 열들을 순회하면서 각 열의 값을 LabelEncoder를 사용하여 숫자 
#형태로 변환하는 작업을 수행한다.
# LabelEncoder()는 Scikit-learn의 클래스로, 범주형 데이터를 숫자형으로 변환하는 역할
#df[col].values는 데이터프레임 df에서 열 col의 값들을 가져오는 것을 의미한다.
#LabelEncoder().fit_transform(df[col].values)는 해당 열의 값을 LabelEncoder를 사용하여 숫자 형태로 변환하는 과정
#fit_transform() 메서드는 LabelEncoder를 피팅(fitting)하고, 해당 열의 값을 변환하여 반환
    
print('******Labeling 후 데이터*****')
print(datasets.head(8))
    
# 상관계수 히트맵(heatmap)
# sns.set(font_scale=1.2)
# sns.set(rc={'figure.figsize':(12, 9)})
# sns.heatmap(
#     data = datasets.corr(),
#     square = True,
#     annot = True,
#     cbar = True
# )
# plt.show()   

x = datasets[['ID', 'Gender', 'Age', 'Region_Code', 'Occupation', 'Channel_Code',
       'Vintage', 'Credit_Product', 'Avg_Account_Balance', 'Is_Active']]
y = datasets[['Is_Lead']]

print(x.shape) # (245725, 10)
print(y.shape) # (245725, 1)

# # 필요없는 칼럼 제거 : Age와 Vintage의 상관계수 0.63으로 가장 높음
# # x = x.drop(['Age', 'Vintage'], axis=1)


feature_name = datasets.feature_names
print(feature_name)

'''
# train 데이터와 test 데이터 분할 -----------------------------------
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    shuffle=True,
    random_state=72
)

#scaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#kfold
n_splits = 5
random_state = 62
kfold = StratifiedKFold(
    n_splits=n_splits,
    random_state=random_state,
    shuffle=True
)

#2. 모델
model = XGBClassifier()

#3. 훈련
model.fit(
    x_train, y_train,
    early_stopping_rounds=20,
    eval_set = [(x_train, y_train), (x_test, y_test)],
    eval_metric = 'merror'
)

#4. 
score = cross_val_score(
    model, 
    x_train, y_train, 
    cv=kfold
)
y_pred = cross_val_predict(
    model, 
    x_test, y_test,
    cv=kfold #cross validation
)

acc = accuracy_score(y_test, y_pred)

print('score :', score)
print('acc :', acc)

# score : [0.95238095 0.9047619  0.95238095 0.9047619  1.        ]
# acc : 0.9333333333333333 // 전체 수행했을 때 일반적인 값

#Select_From_Model
threshold = model.feature_importances_

for thresh in threshold : 
    selection = SelectFromModel(
        model, threshold=thresh, prefit=True
    )
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    
    selection_model = XGBClassifier() #모델은 바뀌어도 된다
    selection_model.fit(select_x_train, y_train)
    y_predict = selection_model.predict(select_x_test)
    score = accuracy_score(y_test, y_predict)
    print("Thresh=%.3f, n=%d, ACC:%.2f%%"%(thresh, select_x_train.shape[1], score*100))
    
    #컬럼명 출력
    selected_feature_indices = selection.get_support(indices=True)
    selected_feature_names = [feature_name[i] for i in selected_feature_indices]
    print(selected_feature_names)
'''