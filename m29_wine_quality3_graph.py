import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
datasets = pd.read_csv('../_data/winequality-white.csv',
                        header=0, sep=';')

# dataset의 바그래프 그리기

print(datasets)
# print(datasets.columns)
# '''
count_data = datasets.groupby('quality')['quality'].count()
print(count_data)
# 라벨을 단순화하면 정확도가 올라갈 수 있다.

plt.bar(count_data.index, count_data)
plt.show()