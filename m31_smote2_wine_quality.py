from imblearn.over_sampling import SMOTE
from sklearn.datasets import load_wine
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import time
import warnings
from sklearn.metrics import f1_score
warnings.filterwarnings('ignore')

datasets = pd.read_csv('../_data/winequality-white.csv',
                        header=0, sep=';')
datasets = datasets.values
x = datasets[:, :11]
y = datasets[:, 11]

print(x.shape, y.shape)
print(pd.Series(y).value_counts())
# 6.0    2198
# 5.0    1457
# 7.0     880
# 8.0     175
# 4.0     163
# 3.0      20
# 9.0       5
print(y)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]

# ##################### 라벨 대통합
print("================================")


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=66, stratify=y
)

print('random state',pd.Series(y_train).value_counts())
# 1    53
# 0    44 -> 53
# 2    14 -> 53
# 모든 라벨의 갯수를 맞춰줘야 함.

model = XGBClassifier(n_jobs=-1)
model.fit(x_train, y_train, eval_metric='mlogloss')

score = model.score(x_test, y_test)
print("model.score", score) # model.score 0.6523809523809524

# '''
########################### smote 적용 ############################
print('============= smote 적용 ==============')

smote = SMOTE(random_state=66, k_neighbors=3)

x_smote_train, y_smote_train= smote.fit_resample(x_train, y_train)

# print(pd.Series(y_smote_train).value_counts())
# print(x_smote_train.shape, y_smote_train.shape) # (159, 13) (159,)

print('smote 전 :', x_train.shape, y_train.shape)
print('smote 후 :', x_smote_train.shape, y_smote_train.shape)
print('smote 전 레이블 값 분포 :\n', pd.Series(y_train).value_counts())
print('smote 후 레이블 값 분포 :\n', pd.Series(y_smote_train).value_counts())

model2 = XGBClassifier(n_jobs=-1)
model2.fit(x_smote_train, y_smote_train, eval_metric='mlogloss')

score = model2.score(x_test, y_test)
print('model2.score :', score) # model2.score : 0.972972972972973
# y_predict = model2.predict(x_test, y_test)
# f1_score = f1_score(y_predict, y_test)
# '''