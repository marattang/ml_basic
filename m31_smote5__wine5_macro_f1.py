'''



3,4,5 -> 0
6 -> 1
7,8,9 -> 2
스모트 하기 전 후 비교하기
더 좋이지는 기준은 macro f1
'''
from imblearn.over_sampling import SMOTE
from sklearn.datasets import load_wine
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import time
import warnings
from sklearn.metrics import f1_score, accuracy_score
import time
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

for i, j in enumerate(y):
    if j == 9 :
        y[i] = 2
    elif j == 8 :
        y[i] = 2
    elif j == 7 :
        y[i] = 2
    elif j == 6:
        y[i] = 6
    elif j == 5:
        y[i] = 0
    elif j == 4:
        y[i] = 0
    elif j == 3:
        y[i] = 0

print('random state',pd.Series(y).value_counts())

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=66, stratify=None
)

# 1    53
# 0    44 -> 53
# 2    14 -> 53
# 모든 라벨의 갯수를 맞춰줘야 함.

model = XGBClassifier(n_jobs=-1)
model.fit(x_train, y_train, eval_metric='mlogloss')

score = model.score(x_test, y_test)
print("model.score", score) # model.score 0.6523809523809524

y_pred = model.predict(x_test)
f1 = f1_score(y_pred, y_test, average='macro')
print('f1 : ', f1)

# '''
########################### smote 적용 ############################
print('============= smote 적용 ==============')
start = time.time()
smote = SMOTE(random_state=66, k_neighbors=60)
x_smote_train, y_smote_train= smote.fit_resample(x_train, y_train)
end = time.time() - start

# print(pd.Series(y_smote_train).value_counts())
# print(x_smote_train.shape, y_smote_train.shape) # (159, 13) (159,)

print('smote 전 :', x_train.shape, y_train.shape)
print('smote 후 :', x_smote_train.shape, y_smote_train.shape)
print('smote 전 레이블 값 분포 :\n', pd.Series(y_train).value_counts())
print('smote 후 레이블 값 분포 :\n', pd.Series(y_smote_train).value_counts())
print('SMOTE 경과 시간 : ', end)
model2 = XGBClassifier(n_jobs=-1)
model2.fit(x_smote_train, y_smote_train, eval_metric='mlogloss')

score = model2.score(x_test, y_test)
print('model2.score :', score) # model2.score : 0.972972972972973
# y_predict = model2.predict(x_test, y_test)
# f1_score = f1_score(y_predict, y_test)
# '''
y_pred = model.predict(x_test)
f1 = f1_score(y_pred, y_test, average='macro')
print('f1 : ', f1)
'''
model.score 0.7068027210884353
f1 :  0.7049595056751449
============= smote 적용 ==============
smote 전 : (3428, 11) (3428,)
smote 후 : (4605, 11) (4605,)
smote 전 레이블 값 분포 :
 6.0    1535
0.0    1166
2.0     727
dtype: int64
smote 후 레이블 값 분포 :
 0.0    1535
6.0    1535
2.0    1535
dtype: int64
SMOTE 경과 시간 :  0.02393627166748047
model2.score : 0.7027210884353742
f1 :  0.7049595056751449
'''