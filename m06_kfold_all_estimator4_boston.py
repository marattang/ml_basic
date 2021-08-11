import numpy as np
from sklearn.utils import all_estimators
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score
import warnings

warnings.filterwarnings('ignore')

datasets = load_boston()

x = datasets.data
y = datasets.target

#2. 모델구성
# 모든 모델을 알고리즘으로 정의. 
allAlgorithms = all_estimators(type_filter='regressor')
# allAlgorithms = all_estimators(type_filter='regressor')
# print(allAlgorithms) # 각종 모델 중 classify인 것들.
print('모델의 갯수 : ',len(allAlgorithms)) 
kfold = KFold(n_splits=5, shuffle=True, random_state=66)

for( name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        scores = cross_val_score(model, x, y, cv=kfold)
        print(name, scores, '평균 : ', round(np.mean(scores), 5))

    except:
        # continue
        print(name, '은 없는 모델')

# 스케일링 방법에 따라 적용할 수 있는 모델이 다름. 어느 스케일링
# 방법을 사용하는지에 따라 모델이 갈린다.
'''
ARDRegression [0.80125693 0.76317071 0.56809285 0.6400258  0.71991866] 평균 :  0.69849
AdaBoostRegressor [0.89695458 0.80539278 0.79915855 0.83387771 0.84875435] 평균 :  0.83683
BaggingRegressor [0.92119231 0.85910638 0.83536852 0.86973298 0.89028365] 평균 :  0.87514
BayesianRidge [0.79379186 0.81123808 0.57943979 0.62721388 0.70719051] 평균 :  0.70377
CCA [0.79134772 0.73828469 0.39419624 0.5795108  0.73224276] 평균 :  0.64712
DecisionTreeRegressor [0.69594484 0.77824896 0.81312729 0.7377668  0.78556763] 평균 :  0.76213
DummyRegressor [-0.00053702 -0.03356375 -0.00476023 -0.02593069 -0.00275911] 평균 :  -0.01351
ElasticNet [0.73383355 0.76745241 0.59979782 0.60616114 0.64658354] 평균 :  0.67077
ElasticNetCV [0.71677604 0.75276545 0.59116613 0.59289916 0.62888608] 평균 :  0.6565
ExtraTreeRegressor [0.69624863 0.74529165 0.66526375 0.68187336 0.75234946] 평균 :  0.70821
ExtraTreesRegressor [0.93825396 0.85590023 0.7682986  0.88098457 0.92890856] 평균 :  0.87447
GammaRegressor [-0.00058757 -0.03146716 -0.00463664 -0.02807276 -0.00298635] 평균 :  -0.01355
GaussianProcessRegressor [-6.07310526 -5.51957093 -6.33482574 -6.36383476 -5.35160828] 평균 :  -5.92859
GradientBoostingRegressor [0.94600966 0.83666561 0.82864346 0.88622172 0.93108541] 평균 :  0.88573
HistGradientBoostingRegressor [0.93235978 0.82415907 0.78740524 0.88879806 0.85766226] 평균 :  0.85808
HuberRegressor [0.74400323 0.64244715 0.52848946 0.37100122 0.63403398] 평균 :  0.584
IsotonicRegression [nan nan nan nan nan] 평균 :  nan
KNeighborsRegressor [0.59008727 0.68112533 0.55680192 0.4032667  0.41180856] 평균 :  0.52862
KernelRidge [0.83333255 0.76712443 0.5304997  0.5836223  0.71226555] 평균 :  0.68537
Lars [0.77467361 0.79839316 0.5903683  0.64083802 0.68439384] 평균 :  0.69773
LarsCV [0.80141197 0.77573678 0.57807429 0.60068407 0.70833854] 평균 :  0.69285
Lasso [0.7240751  0.76027388 0.60141929 0.60458689 0.63793473] 평균 :  0.66566
LassoCV [0.71314939 0.79141061 0.60734295 0.61617714 0.66137127] 평균 :  0.67789
LassoLars [-0.00053702 -0.03356375 -0.00476023 -0.02593069 -0.00275911] 평균 :  -0.01351
LassoLarsCV [0.80301044 0.77573678 0.57807429 0.60068407 0.72486787] 평균 :  0.69647
LassoLarsIC [0.81314239 0.79765276 0.59012698 0.63974189 0.72415009] 평균 :  0.71296
LinearRegression [0.81112887 0.79839316 0.59033016 0.64083802 0.72332215] 평균 :  0.7128
LinearSVR [0.65481132 0.60156866 0.52629811 0.29151405 0.64434993] 평균 :  0.54371
MLPRegressor [0.44972031 0.46301286 0.50682039 0.50259404 0.51234822] 평균 :  0.4869
MultiOutputRegressor 은 없는 모델
MultiTaskElasticNet [nan nan nan nan nan] 평균 :  nan
MultiTaskElasticNetCV [nan nan nan nan nan] 평균 :  nan
MultiTaskLasso [nan nan nan nan nan] 평균 :  nan
MultiTaskLassoCV [nan nan nan nan nan] 평균 :  nan
NuSVR [0.2594254  0.33427351 0.263857   0.11914968 0.170599  ] 평균 :  0.22946
OrthogonalMatchingPursuit [0.58276176 0.565867   0.48689774 0.51545117 0.52049576] 평균 :  0.53429
OrthogonalMatchingPursuitCV [0.75264599 0.75091171 0.52333619 0.59442374 0.66783377] 평균 :  0.65783
PLSCanonical [-2.23170797 -2.33245351 -2.89155602 -2.14746527 -1.44488868] 평균 :  -2.20961
PLSRegression [0.80273131 0.76619347 0.52249555 0.59721829 0.73503313] 평균 :  0.68473
PassiveAggressiveRegressor [-5.2492152  -1.3036133   0.17297447 -1.79659428 -0.51267657] 평균 :  -1.73782
PoissonRegressor [0.85659255 0.8189989  0.66691488 0.67998192 0.75195656] 평균 :  0.75489
RANSACRegressor [0.65474864 0.55084706 0.53035233 0.51765207 0.05845026] 평균 :  0.46241
RadiusNeighborsRegressor [nan nan nan nan nan] 평균 :  nan
RandomForestRegressor [0.9246439  0.85478097 0.81533951 0.88301959 0.90606505] 평균 :  0.87677
RegressorChain 은 없는 모델
Ridge [0.80984876 0.80618063 0.58111378 0.63459427 0.72264776] 평균 :  0.71088
RidgeCV [0.81125292 0.80010535 0.58888304 0.64008984 0.72362912] 평균 :  0.71279
SGDRegressor [-3.77096813e+26 -1.40320397e+25 -1.37200987e+26 -3.55595066e+26
 -2.16038602e+26] 평균 :  -2.1999270132248507e+26
SVR [0.23475113 0.31583258 0.24121157 0.04946335 0.14020554] 평균 :  0.19629
StackingRegressor 은 없는 모델
TheilSenRegressor [0.7931345  0.73189127 0.57442062 0.55807799 0.71620254] 평균 :  0.67475
TransformedTargetRegressor [0.81112887 0.79839316 0.59033016 0.64083802 0.72332215] 평균 :  0.7128
TweedieRegressor [0.7492543  0.75457294 0.56286929 0.57989884 0.63242475] 평균 :  0.6558
VotingRegressor 은 없는 모델
'''