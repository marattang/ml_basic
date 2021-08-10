from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

# print(x.shape, y.shape)
print(x)
print(y)
print(datasets.feature_names)
# ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
print(datasets.DESCR)

print(y[:30])
print(np.min(y), np.max(y))

# case1 연산이 너무 적음 5,4,3,2,1
# case2 노드가 많다가 갑자기 줄어듬. 400, 243, 3, 1
# case3. 1000, 5883, 840, 1233, 1102, 8335 통상적으로 역삼각형 형태가 가장 많음.
# 한 레이어당 지나치게 많은 노드의 수를 배정할 경우 과적합이 일어난다.
'''
model = Sequential()
model.add(Dense(55, input_shape=(10, ), activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(25, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(12, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(1))
'''
model = RandomForestRegressor()
# batch size와 epoch를 둘다 늘릴 경우, 일반화된 데이터의 과적합으로 인해 val_loss가 하락하는 것
#2. 모델구성
x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=9, train_size=0.7)

#3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
# model.fit(x_train, y_train, epochs=220, validation_split=0.1, batch_size=45)
model.fit(x_train, y_train)
# batch size 1 validation split = 0.2 ==> 0.46, 
# batch size 1 validation split = 0.1 ==> 0.49, test_data가 증가하면 validation의 loss값이 미약하지만 줄어든다. 2900까지 감소
# batch size 212 validation split = 0.15, epochs=10000 ==> 3000까지 감소하다가 증가 , 2000epochs 이상부터 2800 ~ 2900 유지 -> 5000부터 과적합 

#4. 평가, 예측
# mse, R2
result = model.score(x_test, y_test)
print('result : ', result)

y_predict = model.predict(x_test)

r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)


# 과제 2
# r2 score :  0.6219390842668302