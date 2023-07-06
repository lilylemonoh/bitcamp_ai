import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# 데이터 불러오기
path = './data/credit_card_prediction/'
data = pd.read_csv(path + 'train.csv')

# 데이터 크기 줄이기
# data = data.sample(frac=0.1, random_state=72)

# 특성과 타겟 변수 분리
x = data[['ID', 'Gender', 'Age', 'Region_Code', 'Occupation', 'Channel_Code',
          'Vintage', 'Credit_Product', 'Avg_Account_Balance', 'Is_Active']]
y = data['Is_Lead']

# 필요없는 열 제거
# x = x.drop(['Age', 'Vintage'], axis=1)

print('******Labeling 전 데이터*****')
print(data.head(11))

# 문자를 숫자로 변경 (LabelEncoder)
df = data.copy()  # data를 복사하여 df로 사용

ob_col = list(df.dtypes[df.dtypes=='object'].index) # object 컬럼 리스트

# NaN 처리
df['Credit_Product'].fillna('Unknown', inplace=True)

for col in ob_col:
    df[col] = LabelEncoder().fit_transform(df[col].values)
    
print('******Labeling 후 데이터*****')
print(df.head(11))

# scaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(df)

# train 데이터와 test 데이터 분할
train_data_ratio = 0.8
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y,
    test_size=1 - train_data_ratio, 
    random_state=415)

# 개별 모델 정의
rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=5)
dt = DecisionTreeClassifier(max_depth=10, min_samples_split=10)

# VotingClassifier 정의
voting_model = VotingClassifier(estimators=[('rf', rf), ('dt', dt)], voting='hard')

# 모델 훈련
voting_model.fit(x_train, y_train)

# 테스트 데이터로 예측 수행
y_pred = voting_model.predict(x_test)

# 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)