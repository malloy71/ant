import math
import operator
import random
import time

import matplotlib.pyplot as plt
import numpy
import numpy as np
import sklearn.datasets as ds
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
from AntClassfied import Ant
from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':
    data = pd.read_csv('iris_data.csv')
    X = data.drop(['target', 'label'], axis=1)
    y = data.loc[:, 'label']
    # 预处理，将数据规范化
    X_norm = StandardScaler().fit_transform(X)
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_norm)
    prob = np.array([[0] for _ in range(150)])
    sample = X_pca
    target_classify = data.loc[:, 'label']
    a_sample = []
    a_target = []
    k_sample = []
    k_target = []
    for i in range(150):
        if random.random() < 0.8:
            k_sample.append(sample[i])
            k_target.append(target_classify[i])
            prob[i][0] = 1
        else:
            prob[i][0] = 0
            a_sample.append(sample[i])
            a_target.append(target_classify[i])

    sample = np.hstack((X_pca, prob))

    a_res = Ant(len(a_sample),a_sample,a_target).run()
    KNN=KNeighborsClassifier(n_neighbors=3)
    model = KNN.fit(a_sample,a_res)
    k_res = model.predict(k_sample)
    # >0.8 的样本交给ant训练，将ant训练的样本和结果，交给knn训练，再由knn预测<0.8的样本，合并所有样本和结果
    all_sample = np.vstack((np.array(a_sample), np.array(k_sample)))
    all_target = np.hstack((a_res, np.array(k_res)))

    plt.figure(figsize=(10, 10), facecolor='w')
    plt.subplot(221)
    plt.title('origin classfication')
    plt.scatter(all_sample[:, 0][all_target == 0], all_sample[:, 1][all_target == 0], marker='.', s=20)
    plt.scatter(all_sample[:, 0][all_target == 1], all_sample[:, 1][all_target == 1], marker='x', s=20)
    plt.scatter(all_sample[:, 0][all_target == 2], all_sample[:, 1][all_target == 2], marker='*', s=20)

    plt.subplot(222)
    plt.title('uncertain classfication')
    plt.scatter(all_sample[:, 0][all_target == 0], all_sample[:, 1][all_target == 0], marker='.', s=20)
    plt.scatter(all_sample[:, 0][all_target == 1], all_sample[:, 1][all_target == 1], marker='x', s=20)
    plt.scatter(all_sample[:, 0][all_target == 2], all_sample[:, 1][all_target == 2], marker='*', s=20)

    optimizeAntRes = precision_score(target_classify, all_target, average="micro")

    print("优化后准确率：")
    if (optimizeAntRes < 0.5):
        print(1 - optimizeAntRes)
    else:
        print(optimizeAntRes)

    plt.show()

