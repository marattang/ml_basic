import numpy as np
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
import numpy as np
from sklearn.preprocessing import MinMaxScaler , StandardScaler, PowerTransformer
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
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
n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=66)

#2. 모델구성. #3. 훈련 #4. 평가 예측
model = LinearSVC()
# acc : [nan nan nan nan nan]

# model = SVC()
# acc : [nan nan nan nan nan]

# model = RandomForestRegressor()
# acc : [0.91808432 0.85684902 0.82598107 0.88433482 0.89990515]

# model = KNeighborsRegressor()
# acc : [0.59008727 0.68112533 0.55680192 0.4032667  0.41180856]

# model = DecisionTreeRegressor()
# acc : [0.80128696 0.69785659 0.77303308 0.70967888 0.72836892]

#3. 컴파일, 훈련
# # 4. 평가, 예측

scores = cross_val_score(model, x, y, cv=kfold)
print('acc :', scores)
print('acc 평균:', round(np.mean(scores), 5))

# results = model.score(x_test, y_test)
# print("model score : ", results)

# y_predict = model.predict(x_test)
# acc = accuracy_score(y_test, y_predict)
# print("accuracy_score : ", acc)

'''
loss = model.evaluate(x_test, y_test)
print(y_test[:5]) # 원래 값
y_predict = model.predict(x_test[:5])
print(y_predict) # 예측 값
'''
# KNeighborsClassifier 0.9111111111111111
# DecisionTreeClassifier 0.8888888888888888
# RandomForestClassifier 0.8888888888888888
# LogisticRegression 0.9777777777777777
# SVC : 0.9555555555555556