# 실습, 모델 구성하고 완료하시오.
# 회귀 데이터를 Classifier로 만들었을 경우에 때려 확인!!

#먹히는지 확인from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

datasets = load_boston()
x = datasets.data
y = datasets.target
'''
model = Sequential()
model.add(Dense(55, input_dim=13))
model.add(Dense(38, activation='relu'))
model.add(Dense(27, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1))
'''
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=5, shuffle=True, test_size=0.3)

model = RandomForestRegressor()
# model.compile(loss="mse", optimizer="adam", loss_weights=1)
# model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.15)
model.fit(x_train, y_train)

result = model.score(x_test, y_test)
print('result : ', result)

y_predict = model.predict(x_test)
print('6의 예측 값 : ', y_predict)

r2 = r2_score(y_test, y_predict)
print('r2 score : ', r2)

# r2 score :  0.8807373045839654

# randomforest regressor
# r2 score :  0.8846710170794806