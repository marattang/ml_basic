# 라벨 0을 112개 삭제

# cancer로 만들기
# 지표는 f1

from imblearn.over_sampling import SMOTE
from sklearn.datasets import load_breast_cancer
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import time
import warnings
import numpy as np
from sklearn.metrics import f1_score
warnings.filterwarnings('ignore')

datasets = load_breast_cancer()
df = pd.DataFrame(np.c_[datasets['data'], datasets['target']],
                  columns= np.append(datasets['feature_names'], ['target']))
df = df.sort_values('target')
df = df.iloc[112:,:]
x = df.iloc[:,:-1]
y = df.iloc[:,-1]

# 1    71
# 0    59
# 2    48
print(y)
# [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
#  0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
#  1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
#  2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]

x_new = x[:-30]
y_new = y[:-30]

print(x_new.shape, y_new.shape)
print(pd.Series(y_new).value_counts())

x_train, x_test, y_train, y_test = train_test_split(
    x_new, y_new, train_size=0.8, shuffle=True, random_state=66, stratify=y_new
)

print('random state',pd.Series(y_train).value_counts())
# 1    53
# 0    44 -> 53
# 2    14 -> 53
# 모든 라벨의 갯수를 맞춰줘야 함.

model = XGBClassifier(n_jobs=-1)
model.fit(x_train, y_train, eval_metric='mlogloss')
score = model.score(x_test, y_test)
print("model.score", score) # model.score 0.9459459459459459
y_pred = model.predict(x_test)
f1 = f1_score(y_pred, y_test, average='macro')
print('f1 : ', f1)
########################### smote 적용 ############################
print('============= smote 적용 ==============')

smote = SMOTE(random_state=66)

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
y_pred = model.predict(x_test)
f1 = f1_score(y_pred, y_test, average='macro')
print('f1 : ', f1)

'''
model.score 1.0
f1 :  1.0
============= smote 적용 ==============
smote 전 : (341, 30) (341,)
smote 후 : (522, 30) (522,)
smote 전 레이블 값 분포 :
 1.0    261
0.0     80
Name: target, dtype: int64
smote 후 레이블 값 분포 :
 0.0    261
1.0    261
Name: target, dtype: int64
model2.score : 1.0
f1 :  1.0
'''