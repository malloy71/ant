import random

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

row = 0
col = 0
# 簇
K = 3
# 权重矩阵
weight = []
# 中心矩阵
mid = []
# 蚂蚁数量
ant_num = 10
# 迭代轮次
interation = 20
# 第i只蚂蚁预测
row_max = []


# 蚂蚁类
class Ant:
    # 信息素矩阵
    information = np.zeros(row, K)
    # 归一化信息素矩阵
    norinformation = np.zeros(row, K)
    # 对于每个样本的预估antSi
    row_max = np.zeros([row, 1])
    # 权重矩阵
    weight = np.zeros([row, K])
    # 中心矩阵
    medium = np.zeros([K, col])
    # 临时适应度函数值
    temp_fitness = 0
    # 当前蚂蚁适应度函数值
    fitness = 100


# 将信息素矩阵归一化
def norInformation(information, norinformation):
    for i in range(row):
        for j in range(K):
            norinformation[i][j] = information[i][j] / (information[i][k] for k in range(k))
    return norinformation


# 蚂蚁（代理）:得出每一只蚂蚁归为第几类
def antS(information, row_max):
    row_max = information.argmax(axis=1)


# 计算权重矩阵
def calWeight(row_max):
    for i in range(row):
        weight[i][row_max[i]] = 1


# 计算中心矩阵
def calMid(ant, data):
    medium = np.dot(weight.reshape(K, row), iris.data)
    for i in range(3):
        num = 0
        for j in range(150):
            if weight[j][i] == 1:
                num += 1
            if num != 0:
                medium[0:4][i] /= num
    return medium


# 计算目标函数值
def calFitness():
    global res
    for j in range(K):
        for i in range(row):
            for v in range(col):
                temp_fitness = res, weight[i][j] * pow((iris.data[i][v] - mid[j][v]), 2)


# 更新信息素矩阵
def updateInformation(temp_fitness, fitness, information):
    rou = 0.2
    if temp_fitness < fitness:
        fitness = temp_fitness
        for i in range(ant_num):
            for j in range(row):
                information[i][j] = (1 - rou) * information[i][j] + 1 / fitness


# 第一轮（准备工作）
def initArgs(information):
    # 初始信息素矩阵
    for i in range(row):
        for j in range(K):
            information[i][j] = np.full((row, K), 0.001, dtype=float)

    # 初始化随机分配样本类别
    for i in range(row):
        row_max[i] = random.randint(1, 4)
    return information, row_max


if __name__ == '__main__':
    # 加载数据集
    iris = load_iris()
    data_shape = np.shape(iris.data)
    row = data_shape[0]
    col = data_shape[1]
    # n只蚂蚁的总表
    S = np.zeros(ant_num, row)
    antList = []
    for s in range(ant_num):
        # 初始化
        ant = Ant()
        # 初始化第一次迭代的信息素和代理
        information, row_max = initArgs(ant.information)
        # 得到所有蚂蚁的对应关系
        for i in range(ant_num):
            for j in range(row):
                S[i][j] = row_max[j]



    # 迭代
    for e in range(interation):
        # 蚂蚁
        for s in range(ant_num):
            if e == 0:
                # 初始化第一次迭代的信息素和代理
                information, row_max = initArgs(ant.information)
                # 得到所有蚂蚁的对应关系
                for i in range(ant_num):
                    for j in range(row):
                        S[i][j] = row_max[j]
                continue
            #从第二次迭代开始
            norinformation = norInformation(information, ant.norinformation)
