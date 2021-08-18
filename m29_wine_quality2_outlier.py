import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
datasets = pd.read_csv('../_data/winequality-white.csv',
                        header=0, sep=';')

def outlier(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])
    print('1사분위 : ', quartile_1) # 2.5
    print('q2 : ', q2) # 6.5
    print('3사분위 : ', quartile_3)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    print('lower_bound',lower_bound)
    print('upper_bound',upper_bound)
    return np.where((data_out>upper_bound) | (data_out<lower_bound))
# 분위값은 데이터에서 딱 떨어지는 수치는 아니고, 위치가 된다. 그렇기 때문에 데이터에 없는 값이 나올 수도 있다.
# outliers_loc = list(map(lambda x : outlier(x), aaa))
outliers_loc = datasets.apply(lambda x: outlier(x))
print('이상치의 위치 : ', outliers_loc)

plt.boxplot(datasets)

plt.show()