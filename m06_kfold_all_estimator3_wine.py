import numpy as np
from sklearn.utils import all_estimators
from sklearn.datasets import load_wine
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

datasets = load_wine()

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
AdaBoostClassifier [0.88888889 0.86111111 0.88888889 0.94285714 0.97142857] 평균 :  0.91063
BaggingClassifier [1.         0.91666667 0.97222222 0.94285714 0.97142857] 평균 :  0.96063
BernoulliNB [0.41666667 0.47222222 0.27777778 0.48571429 0.34285714] 평균 :  0.39905
CalibratedClassifierCV [0.94444444 0.94444444 0.88888889 0.88571429 0.91428571] 평균 :  0.91556
CategoricalNB [       nan        nan        nan 0.94285714        nan] 평균 :  nan
ClassifierChain 은 없는 모델
ComplementNB [0.69444444 0.80555556 0.55555556 0.6        0.6       ] 평균 :  0.65111
DecisionTreeClassifier [0.91666667 0.97222222 0.91666667 0.88571429 0.94285714] 평균 :  0.92683
DummyClassifier [0.41666667 0.47222222 0.27777778 0.48571429 0.34285714] 평균 :  0.39905
ExtraTreeClassifier [0.94444444 0.80555556 0.83333333 0.94285714 0.91428571] 평균 :  0.8881
ExtraTreesClassifier [1.         0.94444444 1.         0.97142857 1.        ] 평균 :  0.98317
GaussianNB [1.         0.91666667 0.97222222 0.97142857 1.        ] 평균 :  0.97206
GaussianProcessClassifier [0.44444444 0.30555556 0.55555556 0.62857143 0.45714286] 평균 :  0.47825
GradientBoostingClassifier [0.97222222 0.91666667 0.88888889 0.97142857 0.97142857] 평균 :  0.94413
HistGradientBoostingClassifier [0.97222222 0.94444444 1.         0.97142857 1.        ] 평균 :  0.97762
KNeighborsClassifier [0.69444444 0.77777778 0.61111111 0.62857143 0.74285714] 평균 :  0.69095
LabelPropagation [0.52777778 0.47222222 0.5        0.4        0.54285714] 평균 :  0.48857
LabelSpreading [0.52777778 0.47222222 0.5        0.4        0.54285714] 평균 :  0.48857
LinearDiscriminantAnalysis [1.         0.97222222 1.         0.97142857 1.        ] 평균 :  0.98873
LinearSVC [0.61111111 0.52777778 0.66666667 0.85714286 0.91428571] 평균 :  0.7154
LogisticRegression [0.97222222 0.94444444 0.94444444 0.94285714 1.        ] 평균 :  0.96079
LogisticRegressionCV [1.         0.94444444 0.97222222 0.94285714 0.97142857] 평균 :  0.96619
MLPClassifier [0.86111111 0.19444444 0.66666667 0.57142857 0.91428571] 평균 :  0.64159
MultiOutputClassifier 은 없는 모델
MultinomialNB [0.77777778 0.91666667 0.86111111 0.82857143 0.82857143] 평균 :  0.84254
NearestCentroid [0.69444444 0.72222222 0.69444444 0.77142857 0.74285714] 평균 :  0.72508
NuSVC [0.91666667 0.86111111 0.91666667 0.85714286 0.8       ] 평균 :  0.87032
OneVsOneClassifier 은 없는 모델
OneVsRestClassifier 은 없는 모델
OutputCodeClassifier 은 없는 모델
PassiveAggressiveClassifier [0.66666667 0.72222222 0.36111111 0.34285714 0.4       ] 평균 :  0.49857
Perceptron [0.61111111 0.80555556 0.47222222 0.48571429 0.62857143] 평균 :  0.60063
QuadraticDiscriminantAnalysis [0.97222222 1.         1.         1.         1.        ] 평균 :  0.99444
RadiusNeighborsClassifier [nan nan nan nan nan] 평균 :  nan
RandomForestClassifier [1.         0.94444444 1.         0.97142857 1.        ] 평균 :  0.98317
RidgeClassifier [1.         1.         1.         0.97142857 1.        ] 평균 :  0.99429
RidgeClassifierCV [1.         1.         1.         0.97142857 1.        ] 평균 :  0.99429
SGDClassifier [0.55555556 0.72222222 0.55555556 0.62857143 0.65714286] 평균 :  0.62381
SVC [0.69444444 0.69444444 0.61111111 0.62857143 0.6       ] 평균 :  0.64571
StackingClassifier 은 없는 모델
VotingClassifier 은 없는 모델
'''