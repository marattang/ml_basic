import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler , StandardScaler, PowerTransformer
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import r2_score, accuracy_score

datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target

# y = to_categorical(y)
# print(y[:5])
# print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=9, train_size=0.7)

# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
# model = Sequential()
# model.add(Dense(128, input_shape=(4, ), activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(3, activation='softmax'))

model = LinearSVC()

#3. 컴파일, 훈련
model.fit(x_train, y_train)
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=300, validation_split=0.2, batch_size=45)

# # 4. 평가, 예측
results = model.score(x_test, y_test)
print("model score : ", results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("accuracy_score : ", acc)

'''
loss = model.evaluate(x_test, y_test)
print(y_test[:5]) # 원래 값
y_predict = model.predict(x_test[:5])
print(y_predict) # 예측 값
'''