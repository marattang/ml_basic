from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

# 1. 데이터
datasets = load_boston()
x = datasets['data']
y = datasets['target']

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, train_size=0.8)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
model = XGBRegressor(n_estimators=2000, learning_rate=0.05, n_jobs=1)

# 3. 훈련
model.fit(x_train, y_train, verbose=1, eval_metric= 'rmse', #,'mae', 'logloss'], 
            eval_set=[(x_train, y_train), (x_test, y_test)],
            early_stopping_rounds=10
)

# dnn verbose처럼 학습되는 걸 보여줌
# 4. 평가
results = model.score(x_test, y_test)
print("results : ", results)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 : ", r2)

# results :  0.9348959737066168
# r2 :  0.9348959737066168

print('================')
hist = evals_result = model.evals_result()
print(hist)

# 그래프 그리기

plt.plot(hist['validation_0']['rmse'])      # x: epoch, y: hist.history['loss']
plt.plot(hist['validation_1']['rmse'])

plt.title('val_0, val_1')
plt.xlabel('epochs')
plt.ylabel('val_0, val_1')
plt.legend(['val_0_rmse', 'val_1_rmse'])
plt.show()