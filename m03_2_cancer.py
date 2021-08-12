from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler, QuantileTransformer, PowerTransformer

datasets = load_breast_cancer()
# 1. 데이터
# 데이터셋 정보 확인
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=66, shuffle=True)

print(x.shape, y.shape) # (input = 30, output = 1)

# y는 0 아니면 1만 있다. 이진분류
print(y[:20])
print(np.unique(y))

# 2. 모델
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# 3. 컴파일, 훈련

model = RandomForestClassifier()
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# metrics에 들어간 건 결과에 반영되지 않고 보여주기만 한다.
# es = EarlyStopping(mode='min', monitor='val_loss', patience=10)
# model.fit(x_train, y_train, batch_size=32, epochs=250, validation_split=0.1, callbacks=[es])
model.fit(x_train, y_train)

# 평가, 예측
result = model.score(x_test, y_test) # evaluate는 loss과 metrics도 반환한다. binary_crossentropy의 loss, accuracy의 loss
# print('loss : ', loss[0])
# print('accuracy : ', loss[1])
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print('acc', acc) # 예측 값