from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler, QuantileTransformer, PowerTransformer
from sklearn.utils import all_estimators
import warnings

warnings.filterwarnings('ignore')

datasets = load_breast_cancer()
# 1. 데이터
# 데이터셋 정보 확인

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=66, shuffle=True)

# 2. 모델
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# 3. 컴파일, 훈련

allAlogorithms = all_estimators(type_filter='classifier')

for (name, algorithm) in allAlogorithms:
    try:
        model = algorithm()

        model.fit(x_train, y_train)

        y_predict = model.predict(x_test)
        acc = accuracy_score(y_predict, y_test)
        print(name, '의 정답률 : ', acc)
    except:

        print(name, '은 없는 모델')

'''
AdaBoostClassifier 의 정답률 :  0.9532163742690059
BaggingClassifier 의 정답률 :  0.9473684210526315
BernoulliNB 의 정답률 :  0.9298245614035088
CalibratedClassifierCV 의 정답률 :  0.9590643274853801
CategoricalNB 은 없는 모델
ClassifierChain 은 없는 모델
ComplementNB 은 없는 모델
DecisionTreeClassifier 의 정답률 :  0.9415204678362573
DummyClassifier 의 정답률 :  0.6432748538011696
ExtraTreeClassifier 의 정답률 :  0.9239766081871345
ExtraTreesClassifier 의 정답률 :  0.9590643274853801
GaussianNB 의 정답률 :  0.9473684210526315
GaussianProcessClassifier 의 정답률 :  0.9707602339181286
GradientBoostingClassifier 의 정답률 :  0.9649122807017544
HistGradientBoostingClassifier 의 정답률 :  0.9707602339181286
KNeighborsClassifier 의 정답률 :  0.9590643274853801
LabelPropagation 의 정답률 :  0.9415204678362573
LabelSpreading 의 정답률 :  0.9415204678362573
LinearDiscriminantAnalysis 의 정답률 :  0.9649122807017544
LinearSVC 의 정답률 :  0.9766081871345029
LogisticRegression 의 정답률 :  0.9883040935672515
LogisticRegressionCV 의 정답률 :  0.9883040935672515
MLPClassifier 의 정답률 :  0.9766081871345029
MultiOutputClassifier 은 없는 모델
MultinomialNB 은 없는 모델
NearestCentroid 의 정답률 :  0.9415204678362573
NuSVC 의 정답률 :  0.9473684210526315
OneVsOneClassifier 은 없는 모델
OneVsRestClassifier 은 없는 모델
OutputCodeClassifier 은 없는 모델
PassiveAggressiveClassifier 의 정답률 :  0.9239766081871345
Perceptron 의 정답률 :  0.9473684210526315
QuadraticDiscriminantAnalysis 의 정답률 :  0.9473684210526315
RadiusNeighborsClassifier 은 없는 모델
RandomForestClassifier 의 정답률 :  0.9707602339181286
RidgeClassifier 의 정답률 :  0.9590643274853801
RidgeClassifierCV 의 정답률 :  0.9590643274853801
SGDClassifier 의 정답률 :  0.9766081871345029
SVC 의 정답률 :  0.9707602339181286
StackingClassifier 은 없는 모델
VotingClassifier 은 없는 모델
'''