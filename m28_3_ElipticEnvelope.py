import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.covariance import EllipticEnvelope

aaa = np.array([[1      ,2      ,10000, 3,     4,      6,       7,      8,   90,     100,    5000],
                [1000   ,2000   ,3,     4000,  5000,   6000,    7000,   8,   9000,   10000,  1001]])

aaa = aaa.transpose()
print(aaa.shape)

outliers = EllipticEnvelope(contamination=.2)
outliers.fit(aaa)

results = outliers.predict(aaa)

print(results)
# 가우스 분산(정규분포)에서 이상값을 탐지하기 위한 개체.
# 공분산을 추정하고, 중심 데이터를 타원형으로 fit하고, central mode를 벗어난 데이터를 제외시킨다.