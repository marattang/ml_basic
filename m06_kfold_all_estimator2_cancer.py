import numpy as np
from sklearn.utils import all_estimators
from sklearn.datasets import load_breast_cancer
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

datasets = load_breast_cancer()

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
AdaBoostClassifier [0.94736842 0.99122807 0.94736842 0.96491228 0.97345133] 평균 :  0.96487
BaggingClassifier [0.96491228 0.92982456 0.94736842 0.94736842 0.94690265] 평균 :  0.94728
BernoulliNB [0.64035088 0.65789474 0.62280702 0.5877193  0.62831858] 평균 :  0.62742
CalibratedClassifierCV [0.89473684 0.93859649 0.89473684 0.92982456 0.97345133] 평균 :  0.92627
CategoricalNB [nan nan nan nan nan] 평균 :  nan
ClassifierChain 은 없는 모델
ComplementNB [0.86842105 0.92982456 0.87719298 0.9122807  0.89380531] 평균 :  0.8963
DecisionTreeClassifier [0.92105263 0.92105263 0.92105263 0.87719298 0.95575221] 평균 :  0.91922
DummyClassifier [0.64035088 0.65789474 0.62280702 0.5877193  0.62831858] 평균 :  0.62742
ExtraTreeClassifier [0.9122807  0.92105263 0.90350877 0.90350877 0.92035398] 평균 :  0.91214
ExtraTreesClassifier [0.96491228 0.97368421 0.95614035 0.93859649 1.        ] 평균 :  0.96667
GaussianNB [0.93859649 0.96491228 0.9122807  0.93859649 0.95575221] 평균 :  0.94203
GaussianProcessClassifier [0.87719298 0.89473684 0.89473684 0.94736842 0.94690265] 평균 :  0.91219
GradientBoostingClassifier [0.94736842 0.96491228 0.95614035 0.93859649 0.98230088] 평균 :  0.95786
HistGradientBoostingClassifier [0.97368421 0.98245614 0.96491228 0.96491228 0.98230088] 평균 :  0.97365
KNeighborsClassifier [0.92105263 0.92105263 0.92105263 0.92105263 0.95575221] 평균 :  0.92799
LabelPropagation [0.36842105 0.35964912 0.4122807  0.42105263 0.38938053] 평균 :  0.39016
LabelSpreading [0.36842105 0.35964912 0.4122807  0.42105263 0.38938053] 평균 :  0.39016
LinearDiscriminantAnalysis [0.94736842 0.98245614 0.94736842 0.95614035 0.97345133] 평균 :  0.96136
LinearSVC [0.92982456 0.93859649 0.92105263 0.72807018 0.94690265] 평균 :  0.89289
LogisticRegression [0.93859649 0.95614035 0.88596491 0.94736842 0.96460177] 평균 :  0.93853
LogisticRegressionCV [0.96491228 0.97368421 0.92105263 0.96491228 0.96460177] 평균 :  0.95783
MLPClassifier [0.90350877 0.95614035 0.90350877 0.94736842 0.90265487] 평균 :  0.92264
MultiOutputClassifier 은 없는 모델
MultinomialNB [0.85964912 0.92105263 0.87719298 0.9122807  0.89380531] 평균 :  0.8928
NearestCentroid [0.86842105 0.89473684 0.85964912 0.9122807  0.91150442] 평균 :  0.88932
NuSVC [0.85964912 0.9122807  0.83333333 0.87719298 0.88495575] 평균 :  0.87348
OneVsOneClassifier 은 없는 모델
OneVsRestClassifier 은 없는 모델
OutputCodeClassifier 은 없는 모델
PassiveAggressiveClassifier [0.81578947 0.90350877 0.85087719 0.9122807  0.97345133] 평균 :  0.89118
Perceptron [0.40350877 0.80701754 0.85964912 0.86842105 0.94690265] 평균 :  0.7771
QuadraticDiscriminantAnalysis [0.93859649 0.95614035 0.93859649 0.98245614 0.94690265] 평균 :  0.95254
RadiusNeighborsClassifier [nan nan nan nan nan] 평균 :  nan
RandomForestClassifier [0.95614035 0.96491228 0.97368421 0.94736842 0.96460177] 평균 :  0.96134
RidgeClassifier [0.95614035 0.98245614 0.92105263 0.95614035 0.95575221] 평균 :  0.95431
RidgeClassifierCV [0.94736842 0.97368421 0.93859649 0.95614035 0.96460177] 평균 :  0.95608
SGDClassifier [0.89473684 0.93859649 0.78947368 0.81578947 0.92920354] 평균 :  0.87356
SVC [0.89473684 0.92982456 0.89473684 0.92105263 0.96460177] 평균 :  0.92099
StackingClassifier 은 없는 모델
VotingClassifier 은 없는 모델
'''