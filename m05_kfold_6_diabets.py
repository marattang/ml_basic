import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler , StandardScaler, PowerTransformer
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import warnings

warnings.filterwarnings('ignore')
datasets = load_diabetes()

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
# model = LinearSVC()
# acc : [0.         0.         0.01136364 0.01136364 0.01136364]

# model = SVC()
# acc : [0.         0.01123596 0.01136364 0.         0.        ]

# model = RandomForestRegressor()
# acc : [0.36543579 0.50286638 0.49058681 0.41091929 0.4267373 ]

# model = KNeighborsRegressor()
# acc : [0.39683913 0.32569788 0.43311217 0.32635899 0.35466969]

model = DecisionTreeRegressor()
# acc : [-0.22027599 -0.15283169 -0.12394496 -0.0296453   0.0583553 ]

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