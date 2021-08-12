import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

(x_train, _), (x_test, _) = mnist.load_data()
# _ <? 빈칸으로 가져온다. 무시하는 것. x_train, x_test만 가져오는 것 만약 x_test만 가져오고 싶으면 x_Test 빼고 '_'
print(x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28)

x = np.append(x_train, x_test, axis=0)
print(x.shape)

# 실습
x = x.reshape(x.shape[0], 28* 28)

pca = PCA(n_components=154)
x = pca.fit_transform(x)
print(x)
print(x.shape)

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)

print(sum(pca_EVR))
cumsum = np.cumsum(pca_EVR)
print(cumsum)
print(np.argmax(cumsum >= 0.95)+1)

# pca를 통해 0.95 이상인거 몇 개? => 154
