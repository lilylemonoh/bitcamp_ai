from sklearn.svm import SVC, LinearSVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

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

#2. 모델
# model = LinearSVC()
model = SVC()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가
result = model.score(x_test, y_test)
print('accuracy', result) # 분류모델은 score = accuracy
#linearSVC result :0.9555555555555556
#SVC result: 0.9777777777777777