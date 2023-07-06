
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import all_estimators

import warnings
warnings.filterwarnings('ignore')

datasets = load_iris()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.3,
    random_state=1234,
    shuffle=True
)

print(x_train.shape, y_train.shape) #(105, 4) (105,)
print(x_test.shape, y_test.shape) # (45, 4) (45,)


# Scaler (정규화) 적용

scaler = MinMaxScaler()
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
AdaBoostClassifier 의 정답률 0.9555555555555556
BaggingClassifier 의 정답률 0.9555555555555556
BernoulliNB 의 정답률 0.3111111111111111
CalibratedClassifierCV 의 정답률 0.9111111111111111
CategoricalNB 의 정답률 0.26666666666666666
ClassifierChain 안 나옴
ComplementNB 의 정답률 0.6222222222222222
DecisionTreeClassifier 의 정답률 0.9777777777777777 *****
DummyClassifier 의 정답률 0.26666666666666666
ExtraTreeClassifier 의 정답률 0.9333333333333333
ExtraTreesClassifier 의 정답률 0.9555555555555556
GaussianNB 의 정답률 0.9555555555555556
GaussianProcessClassifier 의 정답률 0.9555555555555556
GradientBoostingClassifier 의 정답률 0.9555555555555556
HistGradientBoostingClassifier 의 정답률 0.9555555555555556
KNeighborsClassifier 의 정답률 0.9555555555555556
LabelPropagation 의 정답률 0.9555555555555556
LabelSpreading 의 정답률 0.9555555555555556
LinearDiscriminantAnalysis 의 정답률 0.9777777777777777
LinearSVC 의 정답률 0.9333333333333333
LogisticRegression 의 정답률 0.9555555555555556
LogisticRegressionCV 의 정답률 0.9777777777777777
MLPClassifier 의 정답률 0.9555555555555556
MultiOutputClassifier 안 나옴
MultinomialNB 의 정답률 0.6222222222222222
NearestCentroid 의 정답률 0.9555555555555556
NuSVC 의 정답률 0.9555555555555556
OneVsOneClassifier 안 나옴
OneVsRestClassifier 안 나옴
OutputCodeClassifier 안 나옴
PassiveAggressiveClassifier 의 정답률 0.9777777777777777
Perceptron 의 정답률 0.9555555555555556
QuadraticDiscriminantAnalysis 의 정답률 0.9777777777777777
RadiusNeighborsClassifier 의 정답률 0.6222222222222222
RandomForestClassifier 의 정답률 0.9555555555555556
RidgeClassifier 의 정답률 0.8444444444444444
RidgeClassifierCV 의 정답률 0.8222222222222222
SGDClassifier 의 정답률 0.9777777777777777
SVC 의 정답률 0.9555555555555556
StackingClassifier 안 나옴
VotingClassifier 안 나옴
'''
        
        



#linearSVC result :0.9555555555555556
#SVC result: 0.9777777777777777


# StandardScaler 적용 후 accuracy 0.9555555555555556
# MinMaxScaler() 적용 후 accuracy 0.9555555555555556
# MaxAbsScaler() 적용 후 accuracy 0.9555555555555556
# RobustScaler()  적용 후 accuracy 0.9555555555555556