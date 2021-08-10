import numpy as np
from sklearn.utils import all_estimators
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler , StandardScaler, PowerTransformer
from tensorflow.keras.callbacks import EarlyStopping
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

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=66, train_size=0.7)

scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
# 모든 모델을 알고리즘으로 정의. 
allAlgorithms = all_estimators(type_filter='classifier')
# allAlgorithms = all_estimators(type_filter='regressor')
# print(allAlgorithms) # 각종 모델 중 classify인 것들.
print('모델의 갯수 : ',len(allAlgorithms)) 

for( name, algorithm) in allAlgorithms:
    try:
        model = algorithm()

        model.fit(x_train, y_train)

        y_predict = model.predict(x_test)
        acc = accuracy_score(y_predict, y_test)
        print(name, '의 정답률 :',  acc)
    except:
        # continue
        print(name, '은 없는 모델')

# 스케일링 방법에 따라 적용할 수 있는 모델이 다름. 어느 스케일링
# 방법을 사용하는지에 따라 모델이 갈린다.
'''
AdaBoostClassifier 의 정답률 : 0.9333333333333333
BaggingClassifier 의 정답률 : 0.9333333333333333
BernoulliNB 의 정답률 : 0.7333333333333333
CalibratedClassifierCV 의 정답률 : 0.8222222222222222
CategoricalNB 은 없는 모델
ClassifierChain 은 없는 모델
ComplementNB 은 없는 모델
DecisionTreeClassifier 의 정답률 : 0.8888888888888888
DummyClassifier 의 정답률 : 0.28888888888888886
ExtraTreeClassifier 의 정답률 : 0.8888888888888888
ExtraTreesClassifier 의 정답률 : 0.9333333333333333
GaussianNB 의 정답률 : 0.9333333333333333
GaussianProcessClassifier 의 정답률 : 0.9777777777777777
GradientBoostingClassifier 의 정답률 : 0.8888888888888888
HistGradientBoostingClassifier 의 정답률 : 0.9111111111111111
KNeighborsClassifier 의 정답률 : 0.9111111111111111
LabelPropagation 의 정답률 : 0.9111111111111111
LabelSpreading 의 정답률 : 0.9111111111111111
LinearDiscriminantAnalysis 의 정답률 : 1.0
LinearSVC 의 정답률 : 0.8666666666666667
LogisticRegression 의 정답률 : 0.9777777777777777
LogisticRegressionCV 의 정답률 : 1.0
MLPClassifier 의 정답률 : 0.9777777777777777
MultiOutputClassifier 은 없는 모델
MultinomialNB 은 없는 모델
NearestCentroid 의 정답률 : 0.8666666666666667
NuSVC 의 정답률 : 0.9555555555555556
OneVsOneClassifier 은 없는 모델
OneVsRestClassifier 은 없는 모델
OutputCodeClassifier 은 없는 모델
PassiveAggressiveClassifier 의 정답률 : 0.9555555555555556
Perceptron 의 정답률 : 0.9555555555555556
QuadraticDiscriminantAnalysis 의 정답률 : 1.0
RadiusNeighborsClassifier 은 없는 모델
RandomForestClassifier 의 정답률 : 0.8888888888888888
RidgeClassifier 의 정답률 : 0.7777777777777778
RidgeClassifierCV 의 정답률 : 0.7777777777777778
SGDClassifier 의 정답률 : 0.9111111111111111
SVC 의 정답률 : 0.9555555555555556
StackingClassifier 은 없는 모델
VotingClassifier 은 없는 모델
'''