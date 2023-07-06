import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.pipeline import make_pipeline


# 1. 데이터
path = './data/credit_card_prediction/'
datasets = pd.read_csv(path + 'train.csv')

print(datasets.columns)

print('******Labeling 전 데이터*****')
print(datasets.head(11))

# 문자를 숫자로 변경 (LabelEncoder)
df = pd.DataFrame(datasets)

ob_col = list(df.dtypes[df.dtypes=='object'].index) # object 컬럼 리스트

# NaN 처리
df['Credit_Product'].fillna('Unknown', inplace=True)

for col in ob_col:
    df[col] = LabelEncoder().fit_transform(df[col].values)
    
print('******Labeling 후 데이터*****')
print(datasets.head(11))
    
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

x = datasets[['ID', 'Gender', 'Age', 'Region_Code', 'Occupation', 'Channel_Code',
       'Vintage', 'Credit_Product', 'Avg_Account_Balance', 'Is_Active']]
y = datasets[['Is_Lead']]

print(x.shape) # (245725, 10)
print(y.shape) # (245725, 1)

# 필요없는 칼럼 제거 : Age와 Vintage의 상관계수 0.63으로 가장 높음
# x = x.drop(['Age', 'Vintage'], axis=1)

# train 데이터와 test 데이터 분할
train_data_ratio = 0.8
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1-train_data_ratio, random_state=123)

# 결과를 출력하여 분할이 성공적으로 완료되었는지 확인합니다.
print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


cat = CatBoostClassifier()
lgbm = LGBMClassifier()
xgb = XGBClassifier()

# VotingClassifier 정의
model = VotingClassifier(
    estimators=[('cat', cat), ('lgbm', lgbm), ('xgb', xgb)],
    voting='soft',
    n_jobs=-1
)

# 모델을 훈련합니다.
model.fit(x_train, y_train)

# 모델을 사용하여 테스트 세트를 예측하고 정확도를 출력합니다.
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# x = x.drop(['Age', 'Vintage'], axis=1) 주석 처리 후
# Accuracy: 0.8618577678298911

