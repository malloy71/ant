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
# 总代理
antGroup = []
antfit = []


# 蚂蚁类
class Ant:
    # 信息素矩阵
    information = []
    # 归一化信息素矩阵
    norinformation = []
    # 对于每个样本的预估antSi
    row_max = []
    # 权重矩阵
    weight = []
    # 中心矩阵
    medium = []

    def __init__(self):
        self.row_max = np.zeros([row])
        self.weight = np.zeros([row, K])
        self.medium = np.zeros([K, col])
        self.information = np.zeros([row, K])
        self.norinformation = np.zeros([row, K])
        self.fitness = 100

    # 其实可有可无
    # 将信息素矩阵归一化
    def norInformation(self):
        for i in range(row):
            for j in range(K):
                self.norinformation[i][j] = self.information[i][j] / (self.information[i][z] for z in range(3))
        return self.norinformation

    # 蚂蚁（代理）:得出每一个归为第几类，更新 row_max
    def antS(self):
        for r in range(row):
            maxNum = self.information[r][0]
            for c in range(K):
                if self.information[r][c] > maxNum:
                    maxNum = self.information[r][c]
                    self.row_max[r] = c

    # 计算权重矩阵
    def calWeight(self):
        for i in range(row):
            self.weight[i][int(self.row_max[i]) - 1] = 1

    # 计算中心矩阵
    def calMed(self, data):
        self.medium = np.dot(self.weight.reshape(K, row), data)
        for i in range(3):
            num = 0
            for j in range(150):
                if self.weight[j][i] == 1:
                    num += 1
            if num != 0:
                self.medium[0:4][i] /= num

    # 计算目标函数值
    def calFitness(self, data):
        for j in range(K):
            for i in range(row):
                for v in range(col):
                    # TODO 距离公式
                    self.fitness += self.weight[i][j] * pow((data[i][v] - self.medium[j][v]), 2)

    # 更新信息素矩阵
    def updateInformation(self, ant_fitness_sum):

        rou = 0.2

        for i in range(row):
            for j in range(K):
                self.information[i][j] = (1 - rou) * self.information[i][j] + ant_fitness_sum

    # 第一轮（准备工作）
    def initArgs(self):
        # 初始信息素矩阵
        self.information = np.full((row, K), 0.001, dtype=float)

        # 初始化随机分配样本类别
        for i in range(row):
            self.row_max[i] = int(random.randint(1, 3))
    # 选取20位进行交叉
    #
    #     # 定义蚂蚁的输出格式
    #     def __repr__(self):
    #         return "(%s,%s)" % (self.fitness, self.row_max)
    #
    #
    # 交叉


def exchange():
    change_num = 20
    # 随机锁定蚂蚁（第几只和第几只蚂蚁进行交换）
    index1 = 0
    index2 = 0
    # 禁忌表，在同一轮迭代中，交换过的蚂蚁，不能再次交换
    taboo = []
    for i in range(5):
        # 交换index1只蚂蚁与index2只蚂蚁特定位置上的值
        while index2 == index1 or index1 in taboo or index2 in taboo:
            index1 = random.randint(0, ant_num - 1)
            index2 = random.randint(0, ant_num - 1)
        # 将已交换的蚂蚁加入禁忌表中
        taboo.append(index2)
        taboo.append(index1)
        temp_ant1 = antGroup[index1].row_max
        temp_ant2 = antGroup[index2].row_max

        for j in range(change_num):
            index = random.randint(0, row - 1)
            temp_ant1[index], temp_ant2[index] = temp_ant2[index], temp_ant1[index]


# 按照fitness排序
def antCompare(self, ant):
    if (self.fitness > ant.fitness):
        return -1
    else:
        return 1


import functools


# 变异
def change(antGroup, data):
    antGroup = sorted(antGroup, key=functools.cmp_to_key(antCompare))

    # 取最优的两只变异
    for i in range(2):
        ant = antGroup[i]
        for j in range(row):
            if (random.random() > 0.95):
                temp = random.randint(0, K)
                # 若与原先相同则重取
                while temp == tempAnt.row_max[j]:
                    temp = random.randint(0, K)
                tempAnt.row_max[j] = temp
        # 计算tempAnt 的fitness
        # tempAnt.calWeight()
        # tempAnt.calMed(data)
        # # 计算目标函数值
        # tempAnt.calFitness(data)
        # if tempAnt.fitness > ant.fitness:
        #     ant.row_max = tempAnt.row_max


# 计算信息素增量
def updateAnt():
    ant_fitnessSum = 0
    for ant in antGroup:
        ant_fitnessSum += 1 / ant.fitness

    for ant in antGroup:
        ant.updateInformation(ant_fitnessSum)


def run():
    # 加载数据集
    iris = load_iris()
    global row, col, tempAnt
    data_shape = np.shape(iris.data)
    row = data_shape[0]
    col = data_shape[1]
    result = []
    for k in range(interation):
        # 初始化(第一次迭代，创建蚁群)
        if k == 0:
            for i in range(ant_num):
                tempAnt = Ant()
                tempAnt.initArgs()
                antGroup.append(tempAnt)
            for ant in antGroup:
                ant.calWeight()
                ant.calMed(iris.data)
                # 计算目标函数值
                ant.calFitness(iris.data)
            # 随机进行交叉
            change(antGroup, iris.data)
            for ant in antGroup:
                ant.calWeight()
                ant.calMed(iris.data)
                # 计算目标函数值
                ant.calFitness(iris.data)
                # 更新
            updateAnt()
        else:
            # 第二轮迭代往后
            change(antGroup, iris.data)
            for ant in antGroup:
                ant.antS()
                ant.calWeight()
                ant.calMed(iris.data)
                # 计算目标函数值
                ant.calFitness(iris.data)
            # 挑选最优的蚂蚁进行变异
            exchange()
            # 计算目标函数值更新
            for ant in antGroup:
                ant.antS()
                ant.calWeight()
                ant.calMed(iris.data)
                # 计算目标函数值
                ant.calFitness(iris.data)
            updateAnt()
            # 打印这一轮蚁群迭代情况
        perfect_ant = antGroup[0]
        for ant in antGroup:
            if (ant.fitness < perfect_ant.fitness):
                perfect_ant = ant
        # print("第", k, "轮迭代----------")
        print(perfect_ant.row_max)
        result = perfect_ant.row_max
    return result;


if __name__ == '__main__':
    run()
