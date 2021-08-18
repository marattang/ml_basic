import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler

datasets = pd.read_csv('../_data/winequality-white.csv',
                        header=0, sep=';')

print(datasets.head())
print(datasets.shape)
print(datasets.describe())

datasets = datasets.values

x = datasets[:, :11]
y = datasets[:, 11]
print(y.shape)

newlist = []
for i in list(y):
    if i<=4 :
        newlist +=[0]
    elif i<=7:
        newlist +=[1]
    else:
        newlist +=[2]
y = np.array(newlist)
print(y.shape)      # (4898,)

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 모델
model = XGBClassifier()

# 훈련
model.fit(x_train, y_train)

# 평가, 예측
score = model.score(x_test, y_test)

print("accuracy : ", score)
# 라벨 간소화 후
# accuracy :  0.689795918367347 -> accuracy :  accuracy :  0.9469387755102041