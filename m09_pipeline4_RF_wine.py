import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.datasets import load_wine
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
from sklearn.pipeline import make_pipeline, Pipeline
import time

datasets = load_wine()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, train_size=0.7)

#2. 모델구성
model = make_pipeline(MinMaxScaler(), RandomForestClassifier())

#3. 컴파일, 훈련
model.fit(x_train, y_train)

# # 4. 평가, 예측
print("model.score : ", model.score(x_test, y_test))

y_predict = model.predict(x_test)
print('acc : ', accuracy_score(y_predict, y_test))

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