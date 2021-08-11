import numpy as np
from sklearn.utils import all_estimators
from sklearn.datasets import load_iris
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

datasets = load_iris()

x = datasets.data
y = datasets.target

#2. 모델구성
# 모든 모델을 알고리즘으로 정의. 
allAlgorithms = all_estimators(type_filter='classifier')
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
AdaBoostClassifier [0.63333333 0.93333333 1.         0.9        0.96666667] 평균 :  0.88667
BaggingClassifier [0.93333333 0.96666667 1.         0.9        0.96666667] 평균 :  0.95333
BernoulliNB [0.3        0.33333333 0.3        0.23333333 0.3       ] 평균 :  0.29333
CalibratedClassifierCV [0.9        0.83333333 1.         0.86666667 0.96666667] 평균 :  0.91333
CategoricalNB [0.9        0.93333333 0.93333333 0.9        1.        ] 평균 :  0.93333
ClassifierChain 은 없는 모델
ComplementNB [0.66666667 0.66666667 0.7        0.6        0.7       ] 평균 :  0.66667
DecisionTreeClassifier [0.93333333 0.96666667 1.         0.9        0.93333333] 평균 :  0.94667
DummyClassifier [0.3        0.33333333 0.3        0.23333333 0.3       ] 평균 :  0.29333
ExtraTreeClassifier [0.76666667 0.86666667 0.96666667 0.86666667 1.        ] 평균 :  0.89333
ExtraTreesClassifier [0.93333333 0.96666667 1.         0.86666667 0.96666667] 평균 :  0.94667
GaussianNB [0.96666667 0.9        1.         0.9        0.96666667] 평균 :  0.94667
GaussianProcessClassifier [0.96666667 0.96666667 1.         0.9        0.96666667] 평균 :  0.96
GradientBoostingClassifier [0.93333333 0.96666667 1.         0.93333333 0.96666667] 평균 :  0.96
HistGradientBoostingClassifier [0.86666667 0.96666667 1.         0.9        0.96666667] 평균 :  0.94
KNeighborsClassifier [0.96666667 0.96666667 1.         0.9        0.96666667] 평균 :  0.96
LabelPropagation [0.93333333 1.         1.         0.9        0.96666667] 평균 :  0.96
LabelSpreading [0.93333333 1.         1.         0.9        0.96666667] 평균 :  0.96
LinearDiscriminantAnalysis [1.  1.  1.  0.9 1. ] 평균 :  0.98
LinearSVC [0.96666667 0.96666667 1.         0.9        1.        ] 평균 :  0.96667
LogisticRegression [1.         0.96666667 1.         0.9        0.96666667] 평균 :  0.96667
LogisticRegressionCV [1.         0.96666667 1.         0.9        1.        ] 평균 :  0.97333
MLPClassifier [0.96666667 1.         1.         0.93333333 1.        ] 평균 :  0.98
MultiOutputClassifier 은 없는 모델
MultinomialNB [0.96666667 0.93333333 1.         0.93333333 1.        ] 평균 :  0.96667
NearestCentroid [0.93333333 0.9        0.96666667 0.9        0.96666667] 평균 :  0.93333
NuSVC [0.96666667 0.96666667 1.         0.93333333 1.        ] 평균 :  0.97333
OneVsOneClassifier 은 없는 모델
OneVsRestClassifier 은 없는 모델
OutputCodeClassifier 은 없는 모델
PassiveAggressiveClassifier [0.93333333 0.86666667 0.96666667 0.73333333 0.96666667] 평균 :  0.89333
Perceptron [0.66666667 0.66666667 0.93333333 0.73333333 0.9       ] 평균 :  0.78
QuadraticDiscriminantAnalysis [1.         0.96666667 1.         0.93333333 1.        ] 평균 :  0.98
RadiusNeighborsClassifier [0.96666667 0.9        0.96666667 0.93333333 1.        ] 평균 :  0.95333
RandomForestClassifier [0.96666667 0.96666667 1.         0.86666667 0.96666667] 평균 :  0.95333
RidgeClassifier [0.86666667 0.8        0.93333333 0.7        0.9       ] 평균 :  0.84
RidgeClassifierCV [0.86666667 0.8        0.93333333 0.7        0.9       ] 평균 :  0.84
SGDClassifier [0.7        0.83333333 0.8        0.66666667 0.93333333] 평균 :  0.78667
SVC [0.96666667 0.96666667 1.         0.93333333 0.96666667] 평균 :  0.96667
StackingClassifier 은 없는 모델
VotingClassifier 은 없는 모델
'''