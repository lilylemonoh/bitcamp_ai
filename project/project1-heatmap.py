import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# 1. 데이터
path = './data/credit_card_prediction/'
datasets = pd.read_csv(path + 'train.csv')

x = datasets[['ID', 'Gender', 'Age', 'Region_Code', 'Occupation', 'Channel_Code',
       'Vintage', 'Credit_Product', 'Avg_Account_Balance', 'Is_Active']]
y = datasets[['Is_Lead']]

# 필요없는 칼럼 제거
# x = x.drop(['Age', 'Vintage'], axis=1)

# NaN 값 처리
x = x.fillna(x.mode())


# LabelEncoder
ob_col = list(datasets.dtypes[datasets.dtypes=='object'].index) # object 컬럼 리스트
for col in ob_col:
    datasets[col] = LabelEncoder().fit_transform(datasets[col].values)

# train 데이터와 test 데이터 분할
train_data_ratio = 0.8
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1-train_data_ratio, random_state=123)

# 결과를 출력하여 분할이 성공적으로 완료되었는지 확인합니다.
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# 상관계수 히트맵(heatmap)
sns.set(font_scale=1.2)
sns.set(rc={'figure.figsize':(12, 9)})
sns.heatmap(
    data = datasets.corr(),
    square = True,
    annot = True,
    cbar = True
)
plt.show()

'''
# DecisionTree
model = RandomForestClassifier(random_state=42)

# 모델을 훈련합니다.
model.fit(x_train, y_train)

# 모델을 사용하여 테스트 세트를 예측하고 정확도를 출력합니다.
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# DecisionTreeRegressor = Accuracy: 0.7840879031437583
# RandomForestClassifier = Accuracy: 0.8413266863363517
'''
