import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import  XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score


# 데이터 불러오기
path = './data/credit_card_prediction/'
data = pd.read_csv(path + 'train.csv')

# 데이터 크기 줄이기
data = data.sample(frac=0.1, random_state=123)

# 특성과 타겟 변수 분리
x = data[['ID', 'Gender', 'Age', 'Region_Code', 'Occupation', 'Channel_Code',
          'Vintage', 'Credit_Product', 'Avg_Account_Balance', 'Is_Active']]
y = data['Is_Lead']

# 필요없는 열 제거
x = x.drop(['Age', 'Vintage'], axis=1)

# NaN 값 처리
x = x.fillna(x.mode().iloc[0])

# LabelEncoder
ob_col = list(x.dtypes[x.dtypes == 'object'].index)
for col in ob_col:
    x[col] = LabelEncoder().fit_transform(x[col].values)

# scaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# train 데이터와 test 데이터 분할
train_data_ratio = 0.8
x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y,
    test_size=1 - train_data_ratio, 
    random_state=415)

# 개별 모델 정의
nc_model = KNeighborsClassifier(n_neighbors=7)
rf_model = RandomForestClassifier()
dt_model = DecisionTreeClassifier()

cat = CatBoostClassifier()
lgbm = LGBMClassifier()
xgb = XGBClassifier()

# VotingClassifier 정의
model = VotingClassifier(
    #estimators=[('nc_model', nc_model),
    #            ('rf_model', rf_model),
    #            ('dt_model', dt_model)],
    estimators=[('cat', cat), ('lgbm', lgbm), ('xgb', xgb)],
    voting='soft',
    n_jobs=-1
)

# 모델 훈련
model.fit(x_train, y_train)

# 테스트 데이터로 예측 수행
y_pred = model.predict(x_test)

# 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)