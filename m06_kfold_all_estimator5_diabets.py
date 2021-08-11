import numpy as np
from sklearn.utils import all_estimators
from sklearn.datasets import load_diabetes
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

datasets = load_diabetes()

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
ARDRegression [0.49874835 0.48765748 0.56284846 0.37728801 0.53474369] 평균 :  0.49226
AdaBoostRegressor [0.41396308 0.4428625  0.51115115 0.35566173 0.44299261] 평균 :  0.43333
BaggingRegressor [0.34250751 0.3974619  0.39132783 0.27351515 0.37977687] 평균 :  0.35692
BayesianRidge [0.50082189 0.48431051 0.55459312 0.37600508 0.5307344 ] 평균 :  0.48929
CCA [0.48696409 0.42605855 0.55244322 0.21708682 0.50764701] 평균 :  0.43804
DecisionTreeRegressor [-0.24846957 -0.13709161 -0.13742274 -0.07628413  0.06147388] 평균 :  -0.10756
DummyRegressor [-1.54258856e-04 -2.98519672e-03 -1.53442062e-05 -3.80334913e-03
 -9.58335111e-03] 평균 :  -0.00331
ElasticNet [ 0.00810127  0.00637294  0.00924848  0.0040621  -0.00081988] 평균 :  0.00539
ElasticNetCV [0.43071558 0.461506   0.49133954 0.35674829 0.4567084 ] 평균 :  0.4394
ExtraTreeRegressor [-0.10485627 -0.09962904 -0.09997086  0.0224157  -0.15029352] 평균 :  -0.08647
ExtraTreesRegressor [0.3723954  0.48553095 0.52799417 0.42608538 0.45576127] 평균 :  0.45355
GammaRegressor [ 0.00523561  0.00367973  0.0060814   0.00174734 -0.00306898] 평균 :  0.00274
GaussianProcessRegressor [ -5.6360757  -15.27401119  -9.94981439 -12.46884878 -12.04794389] 평균 :  -11.07534
GradientBoostingRegressor [0.39083889 0.48371855 0.48078997 0.39757771 0.44406936] 평균 :  0.4394
HistGradientBoostingRegressor [0.28899498 0.43812684 0.51713242 0.37267554 0.35643755] 평균 :  0.39467
HuberRegressor [0.50334705 0.47508237 0.54650576 0.36883712 0.5173073 ] 평균 :  0.48222
IsotonicRegression [nan nan nan nan nan] 평균 :  nan
KNeighborsRegressor [0.39683913 0.32569788 0.43311217 0.32635899 0.35466969] 평균 :  0.36734
KernelRidge [-3.38476443 -3.49366182 -4.0996205  -3.39039111 -3.60041537] 평균 :  -3.59377
Lars [ 0.49198665 -0.66475442 -1.04410299 -0.04236657  0.51190679] 평균 :  -0.14947
LarsCV [0.4931481  0.48774421 0.55427158 0.38001456 0.52413596] 평균 :  0.48786
Lasso [0.34315574 0.35348212 0.38594431 0.31614536 0.3604865 ] 평균 :  0.35184
LassoCV [0.49799859 0.48389346 0.55926851 0.37740074 0.51636393] 평균 :  0.48699
LassoLars [0.36543887 0.37812653 0.40638095 0.33639271 0.38444891] 평균 :  0.37416
LassoLarsCV [0.49719648 0.48426377 0.55975856 0.37984022 0.51190679] 평균 :  0.48659
LassoLarsIC [0.49940515 0.49108789 0.56130589 0.37942384 0.5247894 ] 평균 :  0.4912
LinearRegression [0.50638911 0.48684632 0.55366898 0.3794262  0.51190679] 평균 :  0.48765
LinearSVR [-0.33470258 -0.31629592 -0.42520583 -0.30276155 -0.47821158] 평균 :  -0.37144
MLPRegressor [-2.79930611 -2.86783774 -3.38406768 -2.85213315 -3.10547732] 평균 :  -3.00176
MultiOutputRegressor 은 없는 모델
MultiTaskElasticNet [nan nan nan nan nan] 평균 :  nan
MultiTaskElasticNetCV [nan nan nan nan nan] 평균 :  nan
MultiTaskLasso [nan nan nan nan nan] 평균 :  nan
MultiTaskLassoCV [nan nan nan nan nan] 평균 :  nan
NuSVR [0.14471275 0.17351835 0.18539957 0.13894135 0.1663745 ] 평균 :  0.16179
OrthogonalMatchingPursuit [0.32934491 0.285747   0.38943221 0.19671679 0.35916077] 평균 :  0.31208
OrthogonalMatchingPursuitCV [0.47845357 0.48661326 0.55695148 0.37039612 0.53615516] 평균 :  0.48571
PLSCanonical [-0.97507923 -1.68534502 -0.8821301  -1.33987816 -1.16041996] 평균 :  -1.20857
PLSRegression [0.47661395 0.4762657  0.5388494  0.38191443 0.54717873] 평균 :  0.48416
PassiveAggressiveRegressor [0.46333479 0.47057229 0.54495554 0.35193283 0.50507701] 평균 :  0.46717
PoissonRegressor [0.32061441 0.35803358 0.3666005  0.28203414 0.34340626] 평균 :  0.33414
RANSACRegressor [ 0.11436238 -0.1640028   0.09651662  0.26492342  0.30398235] 평균 :  0.12316
RadiusNeighborsRegressor [-1.54258856e-04 -2.98519672e-03 -1.53442062e-05 -3.80334913e-03
 -9.58335111e-03] 평균 :  -0.00331
RandomForestRegressor [0.37822918 0.48918578 0.47809966 0.41191089 0.43880177] 평균 :  0.43925
RegressorChain 은 없는 모델
Ridge [0.40936669 0.44788406 0.47057299 0.34467674 0.43339091] 평균 :  0.42118
RidgeCV [0.49525464 0.48761091 0.55171354 0.3801769  0.52749194] 평균 :  0.48845
SGDRegressor [0.39340296 0.44168716 0.4646395  0.32949951 0.41508138] 평균 :  0.40886
SVR [0.14331635 0.18438697 0.17864042 0.1424597  0.1468719 ] 평균 :  0.15914
StackingRegressor 은 없는 모델
TheilSenRegressor [0.50130757 0.4619451  0.55365574 0.34086829 0.52393056] 평균 :  0.47634
TransformedTargetRegressor [0.50638911 0.48684632 0.55366898 0.3794262  0.51190679] 평균 :  0.48765
TweedieRegressor [ 0.00585525  0.00425899  0.00702558  0.00183408 -0.00315042] 평균 :  0.00316
VotingRegressor 은 없는 모델
'''