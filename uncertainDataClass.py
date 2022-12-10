import math
import operator
import random
import time

import matplotlib.pyplot as plt
import numpy
import numpy as np
import sklearn.datasets as ds
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from AntClassfied import Ant


if __name__ == '__main__':
    data = pd.read_csv('iris_data.csv')
    X = data.drop(['target', 'label'], axis=1)
    y = data.loc[:, 'label']
    # 预处理，将数据规范化
    X_norm = StandardScaler().fit_transform(X)
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_norm)
    ablity = np.array([[0] for _ in range(150)])
    sample = X_pca
    target_classify = data.loc[:, 'label']
    a_sample = []
    a_target = []
    for i in range(150):
        if random.random() < 0.8:

            ablity[i][0] = 1
        else :
            ablity[i][0] = 0
            a_sample.append(sample[i])
            a_target.append(target_classify[i])

    sample = np.hstack((X_pca, ablity))
    print(ablity)
    print(sample)
    Ant(len(a_sample),a_sample,a_target).run()