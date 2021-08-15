# 실습
# 분류 -> eval metric을 찾아서 수정from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 데이터
datasets = load_wine()
x = datasets['data']
y = datasets['target']

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, train_size=0.8)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
model = XGBClassifier(n_estimators=500, learning_rate=0.005, n_jobs=1)

# 3. 훈련
model.fit(x_train, y_train, verbose=1, eval_metric= ['auc', 'merror', 'mlogloss'], #,'mae', 'logloss'], 
            eval_set=[(x_train, y_train), (x_test, y_test)])

# dnn verbose처럼 학습되는 걸 보여줌
# 4. 평가
results = model.score(x_test, y_test)
print("results : ", results)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 : ", r2)


# results :  1.0
# r2 :  1.0