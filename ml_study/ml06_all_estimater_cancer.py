from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.utils import all_estimators
import warnings
warnings.filterwarnings('ignore')


# 1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.3,
    random_state=1234,
    shuffle=True
)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# Scaler (정규화) 적용
from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler() # 표준화 scaler
# scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)




#2. 모델
allAlgorithms = all_estimators(type_filter='classifier')
print('allAlgorithms :', allAlgorithms)
print('몇 개? ', len(allAlgorithms)) #41


#3. 출력
for(name, algorithm) in allAlgorithms :
    try :
        model = algorithm()
        model.fit(x_train, y_train)
        result = model.score(x_test, y_test)
        print(name, '의 정답률', result)
    except : 
        print(name, '안 나옴')
        
'''
AdaBoostClassifier 의 정답률 0.9181286549707602
BaggingClassifier 의 정답률 0.9181286549707602
BernoulliNB 의 정답률 0.8947368421052632
CalibratedClassifierCV 의 정답률 0.935672514619883
CategoricalNB 안 나온 놈
ClassifierChain 안 나온 놈
ComplementNB 안 나온 놈
DecisionTreeClassifier 의 정답률 0.9298245614035088
DummyClassifier 의 정답률 0.6140350877192983
ExtraTreeClassifier 의 정답률 0.9005847953216374
ExtraTreesClassifier 의 정답률 0.935672514619883
GaussianNB 의 정답률 0.8888888888888888
GaussianProcessClassifier 의 정답률 0.9239766081871345
GradientBoostingClassifier 의 정답률 0.9181286549707602
HistGradientBoostingClassifier 의 정답률 0.9298245614035088
KNeighborsClassifier 의 정답률 0.9298245614035088
LabelPropagation 의 정답률 0.9239766081871345
LabelSpreading 의 정답률 0.9239766081871345
LinearDiscriminantAnalysis 의 정답률 0.9415204678362573
LinearSVC 의 정답률 0.9532163742690059
LogisticRegression 의 정답률 0.9415204678362573
LogisticRegressionCV 의 정답률 0.935672514619883
MLPClassifier 의 정답률 0.9532163742690059
MultiOutputClassifier 안 나온 놈
MultinomialNB 안 나온 놈
NearestCentroid 의 정답률 0.8888888888888888
NuSVC 의 정답률 0.8888888888888888
OneVsOneClassifier 안 나온 놈
OneVsRestClassifier 안 나온 놈
OutputCodeClassifier 안 나온 놈
PassiveAggressiveClassifier 의 정답률 0.9590643274853801 *****
Perceptron 의 정답률 0.9415204678362573
QuadraticDiscriminantAnalysis 의 정답률 0.9415204678362573
RadiusNeighborsClassifier 안 나온 놈
RandomForestClassifier 의 정답률 0.9298245614035088
RidgeClassifier 의 정답률 0.9415204678362573
RidgeClassifierCV 의 정답률 0.9239766081871345
SGDClassifier 의 정답률 0.9415204678362573
SVC 의 정답률 0.9298245614035088
StackingClassifier 안 나온 놈
VotingClassifier 안 나온 놈
'''






# #3. 훈련
# model.fit(x_train, y_train)

# #4. 평가
# result = model.score(x_test, y_test)
# print('result', result) # 분류모델은 score = accuracy
# #linearSVC result : 0.8771929824561403
# # svc : result 0.8947368421052632


# StandardScaler 적용 후 result 0.9415204678362573
# MinMaxScaler() 적용 후 result 0.9415204678362573
# MaxAbsScaler() 적용 후 result 0.9298245614035088
# RobustScaler()  적용 후 result 0.9298245614035088