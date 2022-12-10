import numpy as np
import pandas as pd
import sklearn.datasets as ds
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = pd.read_csv('iris_data.csv')
X = data.drop(['target', 'label'], axis=1)
y = data.loc[:, 'label']
# 预处理，将数据规范化
X_norm = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_norm)
if __name__ == '__main__':
    print(X_pca,type(X_pca))
    print(X_pca.shape)
