from sklearn.svm import LinearSVC, SVC
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x_data = [0, 0], [0, 1], [1, 0], [1, 1]
y_data = [0, 1, 1, 0]

# 2. 모델
model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))

model.compile(loss='binart_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, batch_size=1, epochs=100)
# 3. 훈련

# 4. 평가, 예측
y_predict = model.predict(x_data)
print(x_data, "의 예측 결과 : ", y_predict)

results = model.evaluate(x_data, y_data)
print('model.score : ', results)

acc = accuracy_score(y_data, y_predict)
print('accuracy_score : ', acc)
