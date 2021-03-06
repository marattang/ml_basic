import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes

aaa = np.array([[1,2,10000,3,4,10000,6,7,8,90,100,5000],
                [1000,2000,3,4000,5000,300,500,8,200,100,100]])
# (2, 10) -> (10, 2)
datasets = load_diabetes()
x = datasets.data
y = datasets.target
df = pd.DataFrame(np.c_[datasets.data, datasets.target], columns=datasets.feature_names + ['target'])
# print(df.columns)
print(aaa.shape)
aaa = aaa.tolist()
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
# outliers_loc = outlier(aaa)
# 리스트
outliers_loc = list(map(lambda x : outlier(x), aaa))

# outliers_loc = df.apply(lambda x: outlier(x))
print('이상치의 위치 : ', outliers_loc[0][0])

# 시각화
# 실습

plt.boxplot(aaa)

plt.show()
# 시각화 시켰을 때 아웃라이어의 비율이 높으면 데이터 자체의 문제가 있을 가능성이 있다. 
# 위 예제에서 보면 10개 데이터 중에서 아웃라이어의 비율이 40%가 넘는다.