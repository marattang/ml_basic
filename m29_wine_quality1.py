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

print(type(datasets))
print(datasets.shape)

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

ros = RandomOverSampler(random_state=42)
x_train, y_train = ros.fit_resample(x_train, y_train)

# 모델
model = XGBClassifier()

# 훈련
model.fit(x_train, y_train)

# 평가, 예측
score = model.score(x_test, y_test)

print("accuracy : ", score)
# accuracy :  0.6816326530612244
# accuracy :  0.689795918367347