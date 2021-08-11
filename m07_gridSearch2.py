# 실습
# m07_1 최적의 파라미터값을 가지고 model을 구성 결과 도출
import numpy as np
from sklearn.metrics import r2_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
import warnings

warnings.filterwarnings('ignore')
datasets = load_iris()

x = datasets.data
y = datasets.target

'''
훈련과 평가 예측을 해야하는데, score 자체가 평가를 하기 때문에 model score에 포함된다.
kFold 5번을 교차검증 한다는 뜻.(n split만큼) 5가 들어가면 20%만큼 test데이터로 잡고, 10이 들어가면 10%만큼 잡음.
그렇기 때문에 x train, test를 분리하지 않은 데이터를 갖고 넣어야 함.
'''
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, train_size=0.7)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=66)

parameters = [
    {'C':[1, 10, 100, 1000], 'kernel':["linear"]}, # 4 * 5 = 20번
    {'C':[1, 10, 100], 'kernel':['rbf'], 'gamma':[0.001, 0.0001]}, # 3 * 1 * 2 * 5 = 30번
    {"C":[1, 10, 100, 1000], "kernel":['sigmoid'], "gamma":[0.001, 0.0001]} # 4 * 1 * 2 * 5 = 40번
]
# 총 90번 돌아가는 모델
#2. 모델구성. #3. 훈련 #4. 평가 예측
model = SVC(C=1, kernel='linear')
# 기존 모델에다가 쓰고싶은 파라미터를 딕셔너리 형태로 정의한다. 기존 모델과
# 넣고 싶은 파라미터들의 형태를 명시해주면 된다.
# 

#3. 컴파일, 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
print("model.score : ", model.score(x_test, y_test))

y_predict = model.predict(x_test)
print('accuracy_score : ', accuracy_score(y_predict, y_test))
'''
최적의 매개변수 :  SVC(C=1, kernel='linear')
best_score :  0.9714285714285715
model.score :  0.9555555555555556
accuracy_score :  0.9555555555555556
'''