from sklearn.svm import LinearSVC, SVC
import numpy as np
from sklearn.metrics import accuracy_score, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x_data = [0, 0], [0, 1], [1, 0], [1, 1]
y_data = [0, 1, 1, 0]

# 2. 모델
# 실습 m02_5파일을 다층 레이어 구성해서 이 파일이 acc=1. 이 나오도록 구성
model = Sequential() 
model.add(Dense(10, input_dim=2, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, batch_size=1, epochs=100)
# 3. 훈련

# 4. 평가, 예측
# y_predict = model.predict(x_data)
# print(x_data, "의 예측 결과 : ", y_predict)

results = model.evaluate(x_data, y_data)
print('model.score : ', results)
y_predict = model.predict(x_data, y_data)
y_predict = np.argmax(y_predict)
r2_score = r2_score(y_data, y_predict)
print('r2_score : ', r2_score)

