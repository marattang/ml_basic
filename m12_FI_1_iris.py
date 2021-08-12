# 실습!!
# 피처임포턴스가 전체 중요도에서 하위 20%미만인 컬럼들을 제거해 데이터셋 재 구성 후
# 각 모델 결과 도출
# 피처 = 컬럼 = 열
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
# 1. 데이터
datasets = load_iris()
df = pd.DataFrame(np.c_[datasets.data, datasets.target], columns=datasets.feature_names + ['target'])
df = df.drop('sepal length (cm)', axis=1)
data = df.to_numpy()
x_train, x_test, y_train, y_test = train_test_split(
    data[:,:-1], data[:,-1], train_size=0.7, random_state=66
    )
# train size 0.8보다 0.7이 좋게 나옴
# 2. 모델
# model = DecisionTreeClassifier()
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
model = XGBClassifier()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가 예측
acc = model.score(x_test, y_test)
print('acc : ', acc)

print(model.feature_importances_) # [0.0125026  0.         0.53835801 0.44913938]
# 트리계열에서는 모델 자체가 성능도 괜찮지만, feature importance라는 기능이 있다. 아이리스는 컬럼이 4개라서 4개 수치가 나온다.
# 4개의 컬럼이 훈련에 대한 영향도 두번째 컬럼같은 경우는 0이 나왔기 때문에 크게 중요하지 않은 컬럼이다. => 절대적이지 않고 상대적
# '의사결정트리'에서 사용했을 때 2번째 컬럼이 크게 도움이 안된다는 얘기
# boosting 계열 쓸려면 xgboost pip install 해주기
'''
def plot_feature_importances_datasets(model):
    n_features = data[:,:-1].shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
            align='center')
    plt.yticks(np.arange(n_features), f_name)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importances_datasets(model)
plt.show()
dict = dict(zip(datasets.feature_names, model.feature_importances_))
dict = sorted(dict.items(), key = lambda item: item[1])
length = int(np.round(len(datasets.feature_names)*0.2))
print(dict[0:length])
'''
# 모든 컬럼 사용, train size 0.8 기준
# model = DecisionTreeClassifier()
# acc :  0.9666666666666667

# model = RandomForestClassifier()
# acc :  0.9333333333333333

# model = GradientBoostingClassifier()
# acc :  0.9333333333333333

# model = XGBClassifier()
# acc :  0.9

# 하위 20% 컬럼 삭제 후
# model = DecisionTreeClassifier()
# acc :  0.9666666666666667

# model = RandomForestClassifier()
# acc :  0.9333333333333333

# model = GradientBoostingClassifier()
# acc :  0.9333333333333333

# model = XGBClassifier()
# acc :  0.9