import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
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
x = x.drop(['ID'], axis=1)

# NaN 값 처리
x = x.fillna(x.mode().iloc[0])

# LabelEncoder를 사용하여 범주형 변수 인코딩
ob_col = list(x.dtypes[x.dtypes == 'object'].index)
for col in ob_col:
    x[col] = LabelEncoder().fit_transform(x[col].values)

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