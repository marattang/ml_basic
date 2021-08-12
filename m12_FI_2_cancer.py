from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

datasets = load_breast_cancer()
# df = pd.DataFrame(np.c_[datasets.data, datasets.target], columns=datasets.feature_names + ['target'])
df = pd.DataFrame(datasets.data, columns=datasets.feature_names)
df = df.drop(['mean radius', 'mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean symmetry'], axis=1)
data = df.to_numpy()
x_train, x_test, y_train, y_test = train_test_split(
    data, datasets.target, train_size=0.8, random_state=66
    )
# 0.7보다 0.8이 좋게 나옴.

model = DecisionTreeClassifier()
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
# model = XGBClassifier()

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
# ['mean radius', 'mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean symmetry']

# 모든 컬럼 사용
# DecisionTreeClassifier
# 0.9298245614035088

# RandomForestClassifier
# 0.9649122807017544

# GradientBoostingClassifier
# 0.956140350877193

# XGBClassifier
# 0.9736842105263158

# 하위 20% 컬럼 제거
# DecisionTreeClassifier
# 0.9298245614035088

# RandomForestClassifier
# 0.9649122807017544

# GradientBoostingClassifier
# 0.956140350877193

# XGBClassifier
# 0.9649122807017544