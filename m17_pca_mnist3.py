import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# _ <? 빈칸으로 가져온다. 무시하는 것. x_train, x_test만 가져오는 것 만약 x_test만 가져오고 싶으면 x_Test 빼고 '_'
print(x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28)

# x = np.append(x_train, x_test, axis=0)
# print(x.shape)

# 실습
x_train = x_train.reshape(60000, 28 * 28)
x_test = x_test.reshape(10000, 28 * 28)

pca = PCA(n_components=None)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

print('x train shape:', x_train.shape)
print('x test shape:', x_test.shape)

# '''
pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)

print(sum(pca_EVR))
cumsum = np.cumsum(pca_EVR)
print(cumsum)
print('count',np.argmax(cumsum >= 1.)+1)
# '''
# pca를 통해 0.95 이상인거 몇 개? => 154

# 1. 모델 구성
# Tensorflow DNN으로 구성하고 기존 Tensorflow DNN 비교
'''
model = Sequential()
model.add(Dense(512, input_shape=(154,), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

# # 3. 컴파일, 훈련       metrics['acc']
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
es = EarlyStopping(mode='min', monitor='val_loss', patience=15)
model.fit(x_train, y_train, epochs=500, batch_size=256, validation_split=0.05, callbacks=[es])

# # 4. 평가, 예측 predict X
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])
print('accuracy : ', loss[1])
'''

# acc로만 평가
# CNN(4차원) -> accuracy :  0.98089998960495 -> 0.991까지 나옴.
# 2차원
# DNN -> accuracy 
# accuracy :  0.9847999811172485
# pca 사용 후
# 154 : accuracy : 0.9805999994277954
# 331 : accuracy : 0.9763000011444092