import numpy as np
from scipy.sparse.construct import random
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

datasets = load_diabetes()
x = datasets.data
y = datasets.target
print(x.shape) # (442, 10)

# n components값을 자유롭게 바꿀 수 있다.
pca = PCA(n_components=10) # n_components의 기준이 명확하지 않음.
x = pca.fit_transform(x)
print(x)
print(x.shape) # (442, 7)

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR) # [0.40242142 0.14923182 0.12059623 0.09554764 0.06621856 0.06027192
#  0.05365605 0.04336832 0.00783199] 압축한 결과에 대한 중요도. 중요한 순서대로 나온다.
# 차원축소 하면 가장 낮은 값을 뺀다. ?
print(sum(pca_EVR))

cumsum = np.cumsum(pca_EVR)
print(cumsum)
print(np.argmax(cumsum >= 0.94)+1)

plt.plot(cumsum)
plt.grid()
plt.show()

# '''
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=66)

model = XGBRegressor()
model.fit(x_train, y_train)

r2 = model.score(x_test, y_test)
print('r2 :', r2)

# pca 전
# r2 : 0.2886786350383085

# pca 후
# r2 : 0.35611682762301655
# 오히려 차원을 축소했을 때 좋아지는 경우가 생길 수 있음. 아닐 수도 있고 ^^;
# feature를 삭제하는 건 아니고, 압축하는 개념.