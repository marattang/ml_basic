# 실습
# 1. 상단모델에 그리드 서치 또는 랜덤서치로 튜닝한 모델 구성
# 

# 2. 위 스레드값으로 SelectFromModel 돌려서
# 최적의 피처 갯수 구할 것

# 3. 위 피처 갯수로 피처 갯수를 조정한 뒤
# 그걸로 다시 랜덤서치 그리드서치해서
# 최적의 R2 구할 것

# 1번 값과 3번 값 비교    # 0.47 이상
from sklearn.utils import all_estimators
from xgboost import XGBRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import numpy as np
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import pandas as pd

# 첫번째, 마지막 컬럼 삭제
# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target
df = pd.DataFrame(np.c_[datasets.data, datasets.target], columns=datasets.feature_names + ['target'])
# print(df.columns)
df = df.drop(['age', 's6'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(
    datasets.data, datasets.target, train_size=0.75, random_state=5
    )
# 2. 모델
# '''
parameters = [{'n_estimators':[50, 80, 100], "learning_rate":[0.1, 0.01],
                "max_depth":[1,5,10], "min_samples_leaf":[3, 11], "base_score":[0.1, 0.2], }
]
n_jobs = -1
# model = GridSearchCV(XGBRegressor(booster='gblinear', normalize_type='forest'), parameters, verbose=1)
model = XGBRegressor(base_score=0.1, colsample_bylevel=None,
             colsample_bynode=None, colsample_bytree=None, gamma=None,
             gpu_id=-1, importance_type='gain', interaction_constraints=None,
             learning_rate=0.1, max_delta_step=None, max_depth=3,
             min_child_weight=None, min_samples_leaf=3,
             monotone_constraints=None, n_estimators=50, n_jobs=8,
             normalize_type='forest', num_parallel_tree=None, random_state=0,
             reg_alpha=0, reg_lambda=0, scale_pos_weight=1, subsample=None,
             tree_method=None, validate_parameters=1, verbosity=None)

# 3. 훈련
model.fit(x_train, y_train)
# '''

# 스케일링 방법에 따라 적용할 수 있는 모델이 다름. 어느 스케일링
# 방법을 사용하는지에 따라 모델이 갈린다.
# '''

# 4. 평가, 예측
# '''
score = model.score(x_test, y_test)
print("model.score : ", score)
# print('최적의 매개변수 : ', model.best_estimator_)
# print('best_params : ', model.best_params_)
# print('best_score : ', model.best_score_)

'''
'''
# y_predict = model.predict(x_test)
# print('r2_score : ', r2_score(y_predict, y_test))
# print('score : ', model.score(x_test, y_test))
print('importance',model.feature_importances_)
thresholds = np.sort(model.feature_importances_)
print('thresholds',thresholds)
# print(thresholds)
# 과적합을 방지하는 방법
# 1. 훈련 데이터 증가
# 2. Dropout, Node수 줄이기
# 3. normalization , regulation, batchNomal - L1, L2
# 4. feature delete 
'''
for thresh in thresholds:
    # print(thresh)
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    # print(select_x_train.shape, select_x_test.shape)
    # idx = np.where(thresholds, thresh)
    # print('idx : ', idx)
    seletion_model = XGBRegressor(n_jobs=-1)
    seletion_model.fit(select_x_train, y_train)

    y_pred = seletion_model.predict(select_x_test)

    score = r2_score(y_test, y_pred)

    print("Thresh=%.3f, n=%d, R2: %.2f%%" 
            %(thresh, select_x_train.shape[1], score*100))
'''
# 1번
# best_score :  0.4614253500889178
# model.score :  0.3357843458999087
# r2_score :  -0.289191212900475

