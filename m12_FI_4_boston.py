from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBClassifier, XGBRFRegressor
import pandas as pd
# '''
datasets = load_boston()
# df = pd.DataFrame(np.c_[datasets.data, datasets.target], columns=datasets.feature_names + ['target'])
df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
df = df.drop(['ZN', 'CHAS', 'RAD'], axis=1)
data = df.to_numpy()
x_train, x_test, y_train, y_test = train_test_split(
    data, datasets.target, train_size=0.8, random_state=66
    )
# train size 0.7보다 0.8이 좋게 나옴
# '''
# datasets = load_boston()
# x_train, x_test, y_train, y_test = train_test_split(datasets.data, datasets.target, train_size=0.8, random_state=66)

# model = DecisionTreeRegressor()
# model = RandomForestRegressor()
# model = GradientBoostingRegressor()
model = XGBRFRegressor()

model.fit(x_train, y_train)
score = model.score(x_test, y_test)

print(score)
print(model.feature_importances_)
'''
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
dict = dict(zip(datasets.feature_names, model.feature_importances_))
dict = sorted(dict.items(), key = lambda item: item[1])
length = int(np.round(len(datasets.feature_names)*0.2))
print(dict[0:length])
'''
# ['ZN', 'CHAS', 'RAD']
# 모든 컬럼 사용
# DecisionTreeClassifier
# 0.7989469097451223

# RandomForestClassifier
# 0.922565825496062

# GradientBoostingClassifier
# 0.9461158607851377

# XGBClassifier
# 0.918604686183674

# 하위 20% 컬럼 제거
# DecisionTreeClassifier
# 0.787873019967597

# RandomForestClassifier
# 0.9241553388623014

# GradientBoostingClassifier
# 0.9449758502281509

# XGBClassifier
# 0.9170321395187058