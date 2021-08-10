# 실습, 모델 구성하고 완료하시오.
# 회귀 데이터를 Classifier로 만들었을 경우에 때려 확인!!

#먹히는지 확인from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import warnings
from sklearn.utils import all_estimators

warnings.filterwarnings('ignore')

datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=5, shuffle=True, test_size=0.3)

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
ARDRegression 의 r2 :  0.5431390725172828
AdaBoostRegressor 의 r2 :  0.79280714724654
BaggingRegressor 의 r2 :  0.8346139824409183
BayesianRidge 의 r2 :  0.5183599527530174
CCA 의 r2 :  0.6694371788158544
DecisionTreeRegressor 의 r2 :  0.7900532830204612
DummyRegressor 의 r2 :  -1.883516824134574e+30
ElasticNet 의 r2 :  0.3557833841438518
ElasticNetCV 의 r2 :  0.2888553280567955
ExtraTreeRegressor 의 r2 :  0.812828788813241
ExtraTreesRegressor 의 r2 :  0.8668732078531547
GammaRegressor 의 r2 :  -7.534067296538297e+30
GaussianProcessRegressor 의 r2 :  -527.905350721072
GradientBoostingRegressor 의 r2 :  0.8877654828855355
HistGradientBoostingRegressor 의 r2 :  0.8137348440361839
HuberRegressor 의 r2 :  0.3753637298208614
IsotonicRegression 은 없는 모델
KNeighborsRegressor 의 r2 :  -0.015413377771031689
KernelRidge 의 r2 :  0.5338324734137234
Lars 의 r2 :  0.46930388735101536
LarsCV 의 r2 :  0.3927132419077598
Lasso 의 r2 :  0.3328118368859989
LassoCV 의 r2 :  0.3904318593464612
LassoLars 의 r2 :  -1.883516824134574e+30
LassoLarsCV 의 r2 :  0.5606777695212668
LassoLarsIC 의 r2 :  0.34322947798020687
LinearRegression 의 r2 :  0.5627774543791305
LinearSVR 의 r2 :  -0.06340158539959884
MLPRegressor 의 r2 :  0.40890133756549485
MultiOutputRegressor 은 없는 모델
MultiTaskElasticNet 은 없는 모델
MultiTaskElasticNetCV 은 없는 모델
MultiTaskLasso 은 없는 모델
MultiTaskLassoCV 은 없는 모델
NuSVR 의 r2 :  -4.6772336356422075
OrthogonalMatchingPursuit 의 r2 :  -0.016752168144117396
OrthogonalMatchingPursuitCV 의 r2 :  0.4793848585203754
PLSCanonical 의 r2 :  0.35150429770789793
PLSRegression 의 r2 :  0.5044851196600064
PassiveAggressiveRegressor 의 r2 :  -0.2616368732698291
PoissonRegressor 의 r2 :  0.6141694858322604
RANSACRegressor 의 r2 :  0.2647142255168383
RadiusNeighborsRegressor 은 없는 모델
RandomForestRegressor 의 r2 :  0.8492057250957717
RegressorChain 은 없는 모델
Ridge 의 r2 :  0.54752161373253
RidgeCV 의 r2 :  0.5603097975242419
SGDRegressor 의 r2 :  -30.65295813953247
SVR 의 r2 :  -5.429201969604989
StackingRegressor 은 없는 모델
TheilSenRegressor 의 r2 :  0.4849579871899138
TransformedTargetRegressor 의 r2 :  0.5627774543791305
TweedieRegressor 의 r2 :  0.33180712474150886
VotingRegressor 은 없는 모델
'''