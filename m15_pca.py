import numpy as np
from scipy.sparse.construct import random
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape) # (442, 10)

# n components값을 자유롭게 바꿀 수 있다.
pca = PCA(n_components=9) # n_components의 기준이 명확하지 않음.
x = pca.fit_transform(x)
print(x)
print(x.shape) # (442, 7)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=66)

model = XGBRegressor()
model.fit(x_train, y_train)

r2 = model.score(x_test, y_test)
print('r2 :', r2)

# pca 전
# r2 : 0.28428734040184866

# pca 후
# r2 : 0.345104152483673

# 원래 컬럼이 10개였는데 9 개로 줄이니까 0.35까지 올라감
