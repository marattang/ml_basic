# 지도
# 회귀 : linear
# 분류 : 

# 비지도
# 클러스터링: K-mean

# 크게는 y값의 유무 차이

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
dataset = load_iris()

irisDF = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
print(irisDF)

kmean = KMeans(n_clusters=3, max_iter=300, random_state=66)
kmean.fit(irisDF)
# 데이콘에서 y라벨값이 없는 데이터를 유추해볼 수 있다.
results = kmean.labels_
print(results)
print(dataset.target)

irisDF['cluster'] = kmean.labels_   # 클러스터링해서 생성한 y값
irisDF['target'] = dataset.target   # 원래 y값
# 위치가 바뀔 수도 있음.
print(dataset.feature_names)

iris_results = irisDF.groupby(['target', 'cluster'])['sepal length (cm)'].count()
print(iris_results)