import math
import operator
import random
import threading
import time

import matplotlib.pyplot as plt
import numpy
import numpy as np
import sklearn.datasets as ds
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

SAMPLE_NUM = 500  # 样本数量
FEATURE_NUM = 2  # 每个样本的特征数量
CLASS_NUM = 2  # 分类数量
ANT_NUM = 200  # 蚂蚁数量
ITERATE_NUM = 30  # 迭代次数
NOW_ITER = 1  # 当前迭代轮次

start=time.perf_counter()

tmp_case, temp_target = ds.make_blobs(250, n_features=2, centers=2, random_state=30)

sample, target_classify = ds.make_blobs(SAMPLE_NUM, n_features=FEATURE_NUM, centers=CLASS_NUM, random_state=99)
# kmeans
model = KMeans(n_clusters=2)
#model = DBSCAN(eps=0.5,min_samples=5,metric='euclidean')
model.fit(tmp_case)
km_res = model.predict(sample)
plt.subplot(223)
plt.title('KMeans classfication')
plt.scatter(sample[:, 0], sample[:, 1], c=km_res, s=30, edgecolors='none')
plt.show()
from sklearn.metrics import precision_score
pre = km_res
optimizeAntRes = precision_score(target_classify, pre)
sc = str(metrics.silhouette_score(sample, pre))

f1 = metrics.f1_score(target_classify, pre)

ch = str(metrics.calinski_harabasz_score(sample, pre))
db = str(metrics.davies_bouldin_score(sample, pre))
if (optimizeAntRes < 0.5):
    print(1 - optimizeAntRes)
else:
    print(optimizeAntRes)
print('sc=' + sc)

print('f1为')
if (f1 < 0.5):
    print(1 - f1)
else:
    print(f1)

print('ch=' + ch)
print('db=' + db)
end = time.perf_counter()
print('time='+str(end-start))
