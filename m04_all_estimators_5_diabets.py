from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import all_estimators
import warnings

warnings.filterwarnings('ignore')
# 1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=9, train_size=0.7)

#2. 모델구성
allAlgorithms = all_estimators(type_filter='regressor')
for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        r2 = r2_score(y_predict, y_test)
        print(name, '의 r2 : ', r2)
    except:
        print(name, '은 없는 모델')

'''
ARDRegression 의 r2 :  0.2837321604046278
AdaBoostRegressor 의 r2 :  -0.2655822284711309
BaggingRegressor 의 r2 :  0.07134907117631006
BayesianRidge 의 r2 :  0.2662868151140044
CCA 의 r2 :  0.10766792126849078
DecisionTreeRegressor 의 r2 :  0.1025639933139807
DummyRegressor 의 r2 :  -6.911711255569144e+30
ElasticNet 의 r2 :  -22505.963867932496
ElasticNetCV 의 r2 :  -0.6268700874683393
ExtraTreeRegressor 의 r2 :  -0.0234922032351097
ExtraTreesRegressor 의 r2 :  0.13445125479209175
GammaRegressor 의 r2 :  -37083.05419073403
GaussianProcessRegressor 의 r2 :  -0.03867913446611748
GradientBoostingRegressor 의 r2 :  0.1940733104924529
HistGradientBoostingRegressor 의 r2 :  0.20888502641826878
HuberRegressor 의 r2 :  0.33213555317094823
IsotonicRegression 은 없는 모델
KNeighborsRegressor 의 r2 :  0.04452995759106082
KernelRidge 의 r2 :  -16.26227421687203
Lars 의 r2 :  0.3001824270230815
LarsCV 의 r2 :  0.24558723267753446
Lasso 의 r2 :  -3.6467445307281574
LassoCV 의 r2 :  0.2540880559290931
LassoLars 의 r2 :  -2.022400917948652
LassoLarsCV 의 r2 :  0.24558723267753446
LassoLarsIC 의 r2 :  0.2662025903059605
LinearRegression 의 r2 :  0.30018242702308195
LinearSVR 의 r2 :  -5284.560441683589
MLPRegressor 의 r2 :  -2131.119697492223
MultiOutputRegressor 은 없는 모델
MultiTaskElasticNet 은 없는 모델
MultiTaskElasticNetCV 은 없는 모델
MultiTaskLasso 은 없는 모델
MultiTaskLassoCV 은 없는 모델
NuSVR 의 r2 :  -44.35480081762611
OrthogonalMatchingPursuit 의 r2 :  -0.926284023623414
OrthogonalMatchingPursuitCV 의 r2 :  0.2976664229654491
PLSCanonical 의 r2 :  0.4500448074216171
PLSRegression 의 r2 :  0.30563180920817845
PassiveAggressiveRegressor 의 r2 :  0.05827230261376526
PoissonRegressor 의 r2 :  -3.5166434208247637
RANSACRegressor 의 r2 :  0.117754457466961
RadiusNeighborsRegressor 의 r2 :  -6.911711255569144e+30
RandomForestRegressor 의 r2 :  0.1150585034290289
RegressorChain 은 없는 모델
Ridge 의 r2 :  -1.3000365789666968
RidgeCV 의 r2 :  0.19099441534908423
SGDRegressor 의 r2 :  -1.276946920929526
SVR 의 r2 :  -32.6925617760434
StackingRegressor 은 없는 모델
TheilSenRegressor 의 r2 :  0.33832779422919323
TransformedTargetRegressor 의 r2 :  0.30018242702308195
TweedieRegressor 의 r2 :  -37073.99988436889
VotingRegressor 은 없는 모델
'''