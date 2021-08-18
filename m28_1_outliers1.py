# 이상치 처리
# 1. 삭제
# 2. Nan처리 후 -> 보간 /// linear
# 3. ....(글측치 처리 방법 유사)
# 확실해보이는 데이터는 데이터를 삭제하지 않는다. 데이터를 수정해서 처리할지, 데이터를 증폭샇지 판단해야하.
# 4. scaler -> RobustScaler, QuantileTransformer
# 5. 모델링 : Tree 계열 DT, RF, XG, LGBM...

import numpy as np

from numpy.core.defchararray import lower

aaa = np.array([1,2,-1000, 4, 6, 7, 8, 90, 100, 500])

# 1사분위, 중위수, 3사분위 Quantile = 분위수
# 3사분위값 - 1사분위값(약 50%) * 1.5까지의 수를 통상적으로 정상 데이터로 본다.

def outlier(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out, [25, 50, 75])
    print('1사분위 : ', quartile_1) # 2.5
    print('q2 : ', q2) # 6.5
    print('3사분위 : ', quartile_3)
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound) | (data_out<lower_bound))
# 분위값은 데이터에서 딱 떨어지는 수치는 아니고, 위치가 된다. 그렇기 때문에 데이터에 없는 값이 나올 수도 있다.
outliers_loc = outlier(aaa)

print('이상치의 위치 : ', outliers_loc)

# 시각화
# 위 데이터를 boxplot으로 그리기
import matplotlib.pyplot as plt

plt.boxplot(aaa)

plt.show()