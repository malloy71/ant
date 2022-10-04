import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import clusteringNew
if __name__ == '__main__':
    iris = load_iris()
    x = iris.data[:, 0]  # X- Axis -sepal length
    y = iris.data[:, 1]  # Y- Axis - sepal length
    species = iris.target  # Species
    x_min, x_max = x.min() - 0.5, x.max() + 0.5
    y_min, y_max = y.min() - 0.5, x.max() + 0.5

    # # Scatterplot
    # plt.figure()
    # plt.title("Iris Dataset - Classification By Sepal Sizes")
    # plt.scatter(x, y, c=species)
    # plt.xlabel('Sepal length')
    # plt.ylabel('Sepal width')
    # plt.xlim(x_min, x_max)
    # plt.ylim(y_min, y_max)
    # plt.xticks(())
    # plt.yticks(())
    # plt.show()

    predict = clusteringNew.run()
    i = 0
    count = 0
    for i in range(len(predict)):
        if(species[i] == predict[i]):
            count+=1
    print((float)(count)/(float)(len(predict)))
    plt.figure()
    plt.title("predict res")
    plt.scatter(x, y, c=predict)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()
