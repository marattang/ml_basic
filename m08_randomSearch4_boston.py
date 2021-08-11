# 실습

# 모델 : RandomForestClassifier
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
import warnings

warnings.filterwarnings('ignore')
datasets = load_boston()

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
    {'n_estimators' : [100, 200]}, # 나무의 수
    {'max_depth' : [6, 8, 10, 12]}, # 트리의 최대 깊이
    {'min_samples_leaf' : [3, 5, 7, 10]}, # 리프 노드에 있어야 하는 최소 샘플 수
    {'min_samples_split' : [2, 3, 5, 10]}, # 내부 노드 분할 시 최소 샘플 수
    {'n_jobs' : [-1, 2, 4]} # 병렬로 실행할 작업 수 
]

# 총 90번 돌아가는 모델
#2. 모델구성. #3. 훈련 #4. 평가 예측
model = GridSearchCV(RandomForestRegressor(), parameters, cv=kfold)
# 기존 모델에다가 쓰고싶은 파라미터를 딕셔너리 형태로 정의한다. 기존 모델과
# 넣고 싶은 파라미터들의 형태를 명시해주면 된다.
# 

#3. 컴파일, 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
print('최적의 매개변수 : ', model.best_estimator_)
print('best_score : ', model.best_score_)
print('best_params : ', model.best_params_)
print("model.score : ", model.score(x_test, y_test))

y_predict = model.predict(x_test)
print('r2 : ', r2_score(y_predict, y_test))
# RandomForest
'''
최적의 매개변수 :  RandomForestRegressor(n_jobs=-1)
best_score :  0.8541078542739451
model.score :  0.8895630153059996
r2 :  0.8703720508206749
'''