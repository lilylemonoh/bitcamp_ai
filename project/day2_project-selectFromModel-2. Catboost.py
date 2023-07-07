import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# 1. 데이터
path = './data/credit_card_prediction/'
datasets = pd.read_csv(path + 'train.csv')

# print(datasets.columns)

# print('******Labeling 전 데이터*****')
# print(datasets.head(11))

# 문자를 숫자로 변경 (LabelEncoder)
df = pd.DataFrame(datasets)

ob_col = list(df.dtypes[df.dtypes=='object'].index) # object 컬럼 리스트


# NaN 처리
df['Credit_Product'].fillna('Unknown', inplace=True)

for col in ob_col:
    df[col] = LabelEncoder().fit_transform(df[col].values)
    
# print('******Labeling 후 데이터*****')
# print(datasets.head(11))
    
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

# print(x.shape) # (245725, 10)
# print(y.shape) # (245725, 1)

# 필요없는 칼럼 제거 : Age와 Vintage의 상관계수 0.63으로 가장 높음
# x = x.drop(['Age', 'Vintage'], axis=1)


feature_name = ['ID', 'Gender', 'Age', 'Region_Code', 'Occupation', 'Channel_Code',
       'Vintage', 'Credit_Product', 'Avg_Account_Balance', 'Is_Active']

# feature_name = x[0]
print(feature_name)

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
# model = XGBClassifier()
model = CatBoostClassifier()

#3. 훈련
model.fit(
    x_train, y_train,
    early_stopping_rounds=20,
    eval_set = [(x_train, y_train), (x_test, y_test)],
    # eval_metric = 'error' # CatBoostClassifier는 ever_metric 제거해야 함
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

# 1. model = XGBClassifier()사용했을 때 일반적인 값-------------------------------------
# score : [0.85826405 0.86224638 0.86029476 0.86169007 0.85968431]
# acc : 0.8550421877967389

# 2. model = CatBoostClassifier() 사용했을 때 일반적인 값-------------------------------------
# score : [0.85864194 0.86146154 0.85971338 0.86325979 0.85956804]
# acc : 0.8565479258797037





#Select_From_Model
threshold = model.feature_importances_

for thresh in threshold : 
    selection = SelectFromModel(
        model, threshold=thresh, prefit=True
    )
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    
    
    
    selection_model = CatBoostClassifier(verbose=0) #Catboost는 verbose=0 추가해야 함
    #모델은 바뀌어도 된다
    selection_model.fit(select_x_train, y_train)
    y_predict = selection_model.predict(select_x_test)
    score = accuracy_score(y_test, y_predict)
    
    print("Thresh=%.3f, n=%d, ACC:%.2f%%"%(thresh, select_x_train.shape[1], score*100))
    
    #컬럼명 출력
    selected_feature_indices = selection.get_support(indices=True)
    selected_feature_names = [feature_name[i] for i in selected_feature_indices]
    print(selected_feature_names)






# 1. model = XGBClassifier()------------------------------------------------------------------------

#     Thresh=0.002, n=9, ACC:85.70%
# ['ID', 'Gender', 'Age', 'Occupation', 'Channel_Code', 'Vintage', 'Credit_Product', 'Avg_Account_Balance', 'Is_Active']
# Thresh=0.003, n=7, ACC:85.71%
# ['Gender', 'Age', 'Occupation', 'Channel_Code', 'Vintage', 'Credit_Product', 'Is_Active']
# Thresh=0.013, n=5, ACC:85.75%
# ['Age', 'Occupation', 'Vintage', 'Credit_Product', 'Is_Active']
# Thresh=0.002, n=10, ACC:85.68%
# ['ID', 'Gender', 'Age', 'Region_Code', 'Occupation', 'Channel_Code', 'Vintage', 'Credit_Product', 'Avg_Account_Balance', 'Is_Active']
# Thresh=0.020, n=4, ACC:85.39%
# ['Occupation', 'Vintage', 'Credit_Product', 'Is_Active']
# Thresh=0.009, n=6, ACC:85.72%
# ['Age', 'Occupation', 'Channel_Code', 'Vintage', 'Credit_Product', 'Is_Active']
# Thresh=0.021, n=3, ACC:84.68%
# ['Vintage', 'Credit_Product', 'Is_Active']
# Thresh=0.898, n=1, ACC:84.41%
# ['Credit_Product']
# Thresh=0.002, n=8, ACC:85.74%
# ['Gender', 'Age', 'Occupation', 'Channel_Code', 'Vintage', 'Credit_Product', 'Avg_Account_Balance', 'Is_Active']
# Thresh=0.029, n=2, ACC:84.41%
# ['Credit_Product', 'Is_Active']


#2. model = CatBoostClassifier()------------------------------------------------------------------------
# Thresh=0.396, n=9, ACC:85.75%
# ['ID', 'Age', 'Region_Code', 'Occupation', 'Channel_Code', 'Vintage', 'Credit_Product', 'Avg_Account_Balance', 'Is_Active']
# Thresh=0.256, n=10, ACC:85.75%
# ['ID', 'Gender', 'Age', 'Region_Code', 'Occupation', 'Channel_Code', 'Vintage', 'Credit_Product', 'Avg_Account_Balance', 'Is_Active']
# Thresh=8.612, n=3, ACC:85.46%
# ['Age', 'Occupation', 'Credit_Product']
# Thresh=0.449, n=8, ACC:85.79%
# ['Age', 'Region_Code', 'Occupation', 'Channel_Code', 'Vintage', 'Credit_Product', 'Avg_Account_Balance', 'Is_Active']
# Thresh=25.527, n=2, ACC:84.47%
# ['Occupation', 'Credit_Product']
# Thresh=2.018, n=5, ACC:85.60%
# ['Age', 'Occupation', 'Channel_Code', 'Vintage', 'Credit_Product']
# Thresh=8.080, n=4, ACC:85.66%
# ['Age', 'Occupation', 'Vintage', 'Credit_Product']
# Thresh=52.542, n=1, ACC:84.41%
# ['Credit_Product']
# Thresh=0.552, n=7, ACC:85.77%
# ['Age', 'Occupation', 'Channel_Code', 'Vintage', 'Credit_Product', 'Avg_Account_Balance', 'Is_Active']
# Thresh=1.569, n=6, ACC:85.75%
# ['Age', 'Occupation', 'Channel_Code', 'Vintage', 'Credit_Product', 'Is_Active']