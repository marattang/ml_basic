# [1, np.nan, np.nan, 8, 10]

#  관측치 처리
# 1. 행 삭제
# 2. 0을 넣는다 -> 특정값 -> [1, 0, 0, 8, 10]
# 3. 앞의 값                 [1, 1, 1, 8, 10]
# 4. 뒤의 값                 [1, 8, 8, 8, 10]
# 5. 중위값                  [1, 4.5, 4.5, 8, 10]
# 6. 보간
# 7. 모델링 - predict 
# 8. 부스트계열은 결측치에 대해 자유(?)롭다. -> Tree 계열 DT, RF, XG, LGBM...

from datetime import datetime
import numpy as np
import pandas as pd

# 이상치를 먼저 제거하고 결측치를 채울지, 결측치를 채우고 이상치를 제거할지는 딱히 정해진 건 없다.
datestrs = ['8/13/2021', '8/14/2021', '8/15/2021', '8/16/2021', '8/17/2021']
dates = pd.to_datetime(datestrs)
print(dates)
print(type(dates))  # <class 'pandas.core.indexes.datetimes.DatetimeIndex'>

ts = pd.Series([1, np.nan, np.nan, 8, 10], index=dates)
print('ts : ', ts)

ts_intp_linear = ts.interpolate()
# 보간법은 linear를 기준으로 둔다.
print(ts_intp_linear)


