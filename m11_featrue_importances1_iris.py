# 피처 = 컬럼 = 열
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np 

# 1. 데이터
datasets = load_iris()
x_train, x_test, y_train, y_test = train_test_split(
    datasets.data, datasets.target, train_size=0.8, random_state=66
    )

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

def plot_feature_importances_datasets(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
            align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importances_datasets(model)
plt.show()