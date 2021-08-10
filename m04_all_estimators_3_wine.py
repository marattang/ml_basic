from re import A
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.utils import all_estimators
import warnings

warnings.filterwarnings('ignore')

dataset = load_wine()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, shuffle=True, random_state=66)

scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# 
allAlgorithmns = all_estimators(type_filter='classifier')

for (name, algorithm) in allAlgorithmns:
    try:
        model = algorithm()
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_predict, y_test)
        print(name, '의 정답률 : ', acc)
    except:
        print(name, '은 없는 모델입니다.')

'''
AdaBoostClassifier 의 정답률 :  0.5370370370370371
BaggingClassifier 의 정답률 :  0.9814814814814815
BernoulliNB 의 정답률 :  0.9444444444444444
CalibratedClassifierCV 의 정답률 :  0.9814814814814815
CategoricalNB 은 없는 모델입니다.
ClassifierChain 은 없는 모델입니다.
ComplementNB 은 없는 모델입니다.
DecisionTreeClassifier 의 정답률 :  0.9629629629629629
DummyClassifier 의 정답률 :  0.4074074074074074
ExtraTreeClassifier 의 정답률 :  0.8518518518518519
ExtraTreesClassifier 의 정답률 :  1.0
GaussianNB 의 정답률 :  0.9814814814814815
GaussianProcessClassifier 의 정답률 :  0.9629629629629629
GradientBoostingClassifier 의 정답률 :  0.9629629629629629
HistGradientBoostingClassifier 의 정답률 :  1.0
KNeighborsClassifier 의 정답률 :  0.9259259259259259
LabelPropagation 의 정답률 :  0.9629629629629629
LabelSpreading 의 정답률 :  0.9629629629629629
LinearDiscriminantAnalysis 의 정답률 :  0.9814814814814815
LinearSVC 의 정답률 :  0.9629629629629629
LogisticRegression 의 정답률 :  1.0
LogisticRegressionCV 의 정답률 :  1.0
MLPClassifier 의 정답률 :  1.0
MultiOutputClassifier 은 없는 모델입니다.
MultinomialNB 은 없는 모델입니다.
NearestCentroid 의 정답률 :  0.9629629629629629
NuSVC 의 정답률 :  0.9814814814814815
OneVsOneClassifier 은 없는 모델입니다.
OneVsRestClassifier 은 없는 모델입니다.
OutputCodeClassifier 은 없는 모델입니다.
PassiveAggressiveClassifier 의 정답률 :  0.9814814814814815
Perceptron 의 정답률 :  0.9629629629629629
QuadraticDiscriminantAnalysis 의 정답률 :  1.0
RadiusNeighborsClassifier 은 없는 모델입니다.
RandomForestClassifier 의 정답률 :  1.0
RidgeClassifier 의 정답률 :  0.9814814814814815
RidgeClassifierCV 의 정답률 :  1.0
SGDClassifier 의 정답률 :  1.0
SVC 의 정답률 :  0.9814814814814815
StackingClassifier 은 없는 모델입니다.
VotingClassifier 은 없는 모델입니다.
'''