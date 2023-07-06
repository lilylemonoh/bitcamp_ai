import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 데이터 불러오기
path = './data/credit_card_prediction/'

### train data 불러오기
train_data = pd.read_csv(path + 'train.csv')

# test data 불러오기 
test_data = pd.read_csv(path + 'test.csv')

print(train_data.info())
print(test_data.info())

print(train_data.shape) #(245725, 11)
print(test_data.shape) # (105312, 10)



# 정수형 열
num_cols = ['Age', 'Vintage', 'Avg_Account_Balance']
# 범주형 열
cat_cols = ['Gender', 'Region_Code', 'Occupation', 'Channel_Code', 'Credit_Product', 'Is_Active']
#Target
target = 'Is_Lead'


# Label Encoding the categorical features 

## Custom fuction for label encoding 

def df_lbl_enc(df):
    for c in cat_cols:
        lbl = LabelEncoder()
        df[c] = lbl.fit_transform(df[c])
    return df

## Label Encoding Categorical Columns in train data 

train_data = df_lbl_enc(train_data)

## Label Encoding Categorical Columns in test data 

test_data = df_lbl_enc(test_data)


# ### Train/Test Split 

## Preparing Train data 
## Dropping few columns
x_train = train_data.drop(['ID', 'Gender','Is_Lead','Vintage', 'Avg_Account_Balance'], axis=1)
y_train = train_data[target].values
## Preparing Test data 
## Dropping few columns
ID = test_data['ID']
X_test = test_data.drop(['ID','Gender', 'Vintage', 'Avg_Account_Balance'], axis=1)

model = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10],
    'min_samples_split': [5, 10]
}

grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(x_train, y_train)



'''
# 데이터 크기 줄이기
data = data.sample(frac=0.1, random_state=123)

# 특성과 타겟 변수 분리
x = data[['ID', 'Gender', 'Age', 'Region_Code', 'Occupation', 'Channel_Code',
          'Vintage', 'Credit_Product', 'Avg_Account_Balance', 'Is_Active']]
y = data['Is_Lead']

# 필요없는 열 제거
x = x.drop(['ID'], axis=1)

# NaN 값 처리
x = x.fillna(x.mode().iloc[0])



# 특성 스케일링
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# 훈련 데이터와 테스트 데이터 분할
train_data_ratio = 0.8
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=1 - train_data_ratio, random_state=123)

# 모델 훈련 및 하이퍼파라미터 튜닝
model = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10],
    'min_samples_split': [5, 10]
}

grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(x_train, y_train)

best_model = grid_search.best_estimator_

# 테스트 데이터로 예측 수행
y_pred = best_model.predict(x_test)

# 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
'''