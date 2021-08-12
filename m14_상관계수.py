import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

datasets = load_iris()
print(datasets.keys())
# dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])

print(datasets.target_names)
# ['setosa' 'versicolor' 'virginica'] y 값의 target name, 0,1,2 원래 이름

x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (150, 4) (150,)

df = pd.DataFrame(x, columns=datasets.feature_names)
print(df)

# y 컬럼 추가
df['Target'] = y
print(df)

print("================== 상관계수 히트 맵 ===================")
print(df.corr())

sns.set(font_scale=1.2)
# pearson : standard correlation coefficient 
# kendall : Kendall Tau correlation coefficient
# spearman : Spearman rank correlation
sns.heatmap(data=df.corr(method='pearson'), square=True)

plt.show()

'''
상관분석은 연속적인 두 변수의 상관관계를 분석한다. 
등간척도 이상으로 측정되는 두 변수의 상관관계를 측정하는 데에는 피어슨 상관계수가 쓰이고
서열척도인 두 변수들의 상관관계를 측정하는 데에는 스피어만 상관계수가 쓰인다.
-1, 1 ~ -0.9, 0.9 : 매우 강함
-0.8, 0.8 ~ -0.7, 0.7 : 강함
-0.6, 0.6 ~ -0.4, 0.4 : 상관관계가 있음
-0.3, 0.3 ~ -0.2, 0.2 : 약함
-0.1, 0.1 ~ -0, 0 : 매우 약함
모수적 상관계수에는 가장 많이 알고 있는 피어슨 상관계수를 사용한다.
비모수적 상관계수로는 스피어만 상관계수, 켄달 검정이 있다.
'''