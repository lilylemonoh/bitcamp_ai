import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
import time

#optuna- 최적의 파라미터를 찾아주기 위한 것

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.7,
    random_state=72,
    shuffle=True
)

# Scaler (정규화) 적용
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

# optuna 적용 
import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error

def objectiveCAT(trial: Trial, x_train, y_train, x_test): 
    param = {
        'n_estimators' : trial.suggest_int('n_estimators', 500, 4000), 
        'depth' : trial.suggest_int('depth', 1, 16),
        'fold_permutation_block' : trial.suggest_int('fold_permutation_block', 1, 256), 
        'learning_rate' : trial.suggest_float('learning_rate', 0, 1),
        'od_pval' : trial.suggest_float('od_pval', 0, 1),
        'l2_leaf_reg' : trial.suggest_float('l2_leaf_reg', 0, 4),
        'random_state' :trial.suggest_int('random_state', 1, 2000) 
    }
    # 학습  모델  생성
    model = CatBoostClassifier(**param)
    CAT_model = model.fit(x_train, y_train, verbose=True) # 학습  진행 
    # 모델  성능  확인
    score = accuracy_score(CAT_model.predict(x_test), y_test) 
    return score

# MAE가  최소가  되는  방향으로  학습을  진행
# TPESampler : Sampler using TPE (Tree-structured Parzen Estimator) algorithm. 
study = optuna.create_study(direction='maximize', sampler=TPESampler())
# n_trials 지정해주지  않으면, 무한  반복
study.optimize(lambda trial : objectiveCAT(trial, x, y, x_test), n_trials = 5) 
print('Best trial : score {}, /nparams {}'.format(study.best_trial.value,
study.best_trial.params))