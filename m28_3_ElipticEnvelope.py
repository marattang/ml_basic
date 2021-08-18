import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes


aaa = np.array([[1      ,2      ,10000, 3,     4,      10000,       6,      7,      8,   90,     100,    5000],
                [1000   ,2000   ,3,     4000,  5000,   6000,        7000,   8,      9000,   10000,  1001]])

aaa = aaa.transpose()
print(aaa.shape)

