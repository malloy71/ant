import math
import operator
import random
import threading
import time

import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets as ds

SAMPLE_NUM = 50  # 样本数量
FEATURE_NUM = 2  # 每个样本的特征数量
CLASS_NUM = 2  # 分类数量
ANT_NUM = 200  # 蚂蚁数量
ITERATE_NUM = 200  # 迭代次数

"""
初始化测试样本，sample为样本，target_classify为目标分类结果用于对比算法效果
"""
sample, target_classify = ds.make_blobs(SAMPLE_NUM, n_features=FEATURE_NUM, centers=CLASS_NUM, random_state=3)

"""
信息素矩阵
"""
tao_array = [[random.random() for col in range(FEATURE_NUM)] for row in range(SAMPLE_NUM)]

"""
蚁群解集
"""
ant_array = [[0 for col in range(SAMPLE_NUM)] for row in range(ANT_NUM)]

t_ant_array = [[0 for col in range(SAMPLE_NUM)] for row in range(ANT_NUM)]  # 存储局部搜索时的临时解

"""
聚类中心点
"""
center_array = [[0 for col in range(FEATURE_NUM)] for row in range(CLASS_NUM)]

"""
当前轮次蚂蚁的目标函数值，前者是蚂蚁编号、后者是目标函数值
"""
ant_target = [(0, 0) for col in range(ANT_NUM)]  # 生成ANT_NUM个（0，0）

change_q = 0.3  # 更新蚁群时的转换规则参数，表示何种比例直接根据信息素矩阵进行更新
L = int(ANT_NUM / 5)  # 局部搜索的蚂蚁数量
change_jp = 0.3  # 局部搜索时该样本是否变动
change_rho = 0.02  # 挥发参数
Q = 0.1  # 信息素浓度参数


def _init_test_data(r):
    """
    初始化蚁群解集，随机确认每只蚂蚁下每个样本的分类为1或者0
    """
    for i in range(0, ANT_NUM):
        for j in range(0, SAMPLE_NUM):
            tmp = random.randint(0, FEATURE_NUM - 1)

            ant_array[i][j] = tmp

    """
    将前两个样本作为聚类中心点的初始值
    """
    original_init_center()
    # pick_center_by_density(r)


# 随机选取两个中心点
def original_init_center():
    for i in range(0, CLASS_NUM):
        # 两个属性，i个类
        center_array[i][0] = sample[random.randint(0, SAMPLE_NUM - 1)][0]
        center_array[i][1] = sample[random.randint(0, SAMPLE_NUM - 1)][1]


# 改进后
def change_init_test_data():
    """
    根据初始聚类中心，建立信息素矩阵
    """
    pick_center_by_density()
    for i in range(SAMPLE_NUM):
        for j in range(CLASS_NUM):
            dist = [[0 for col in range(CLASS_NUM)] for row in range(SAMPLE_NUM)]
            dist[i][j] = cal_dis(sample[i], center_array[j])
            tao_array[i][j] = 1 / (CLASS_NUM * dist[i][j])


# 改动点1：根据密度选取中心点
def pick_center_by_density(r):
    # 半径
    # r = 2
    # 每个样本的密度
    density_arr = [0 for col in range(SAMPLE_NUM)]
    # 样本之间的距离矩阵
    dis_arr = [[0 for col in range(SAMPLE_NUM)] for row in range(SAMPLE_NUM)]
    for i in range(SAMPLE_NUM):
        for j in range(SAMPLE_NUM):
            if i == j:
                continue
            dis = cal_dis(sample[i], sample[j])
            dis_arr[i][j] = dis
            if dis <= r:
                density_arr[i] += 1
    # 待改进，综合考虑密度与距离
    for i in range(0, CLASS_NUM):
        max_index = findMax(density_arr, dis_arr, r)
        center_array[i][0] = sample[max_index][0]
        center_array[i][1] = sample[max_index][1]


def cal_dis(param, param1):
    x1 = param[0]
    y1 = param[1]
    x2 = param1[0]
    y2 = param1[1]
    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))


def findMax(density_arr, dis_arr, r):
    index1 = 0
    index2 = 0
    while (dis_arr[index1][index2] < r):
        for i in range(len(density_arr)):
            if density_arr[i] > density_arr[index1]:
                index1 = i
        density_arr[index1] = 0
        for i in range(len(density_arr)):
            if density_arr[i] > density_arr[index2]:
                index2 = i
        density_arr[index2] = 0

    return (index1, index2)


def _get_best_class_by_tao_value(sampleid):
    max_value = np.max(tao_array[sampleid])

    for i in range(0, CLASS_NUM):

        if max_value == tao_array[sampleid][i]:
            return i


def random_pick(some_list, probabilities):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability: break
    return item


def _get_best_class_by_tao_probablity(sampleid):
    tarray = np.array(tao_array[sampleid])

    parray = tarray / np.sum(tarray)

    return random_pick([0, 1], parray)


def global_optimize():
    # 产生一个随机数矩阵
    r = np.random.random((ANT_NUM, SAMPLE_NUM))

    for i in range(0, ANT_NUM):
        for j in range(0, SAMPLE_NUM):

            if r[i][j] > change_q:

                tmp_index = _get_best_class_by_tao_value(j)

                # 选择该样本中信息素最高的做为分类
                ant_array[i][j] = tmp_index

            else:
                # 计算概率值，根据概率的大小来确定一个选项
                tmp_index = _get_best_class_by_tao_probablity(j)

                ant_array[i][j] = tmp_index


def change_global_optimize():
    # 遗传选择交叉
    for i in range(0, ANT_NUM):
        for j in range(0, SAMPLE_NUM):
            pass


def _update_ant():
    """
    更新蚁群步骤
    """
    # 原全局
    global_optimize()
    # 优化后全局
    # _global_search()

    # print(ant_array[i])
    # 1. 确定一个新的聚类中心
    f_value_feature_0 = 0
    f_value_feature_1 = 0

    for i in range(0, CLASS_NUM):

        f_num = 0

        for j in range(0, ANT_NUM):

            for k in range(0, SAMPLE_NUM):

                if ant_array[j][k] == 0:

                    f_num += 1
                    f_value_feature_0 += sample[k][0]  # 特征1

                else:

                    f_num += 1
                    f_value_feature_1 += sample[k][1]  # 特征2

        if i == 0:
            center_array[i][0] = f_value_feature_0 / f_num
        else:
            center_array[i][1] = f_value_feature_1 / f_num

        # print(center_array[i], f_num)


def _judge_sample(sampleid):
    """
    计算与当前聚类点的举例，判断该sample应所属的归类
    """
    target_value_0 = 0
    target_value_1 = 0

    f1 = math.pow((sample[sampleid][0] - center_array[0][0]), 2)
    f2 = math.pow((sample[sampleid][1] - center_array[0][1]), 2)
    target_value_0 = math.sqrt(f1 + f2)

    f1 = math.pow((sample[sampleid][0] - center_array[1][0]), 2)
    f2 = math.pow((sample[sampleid][1] - center_array[1][1]), 2)
    target_value_1 = math.sqrt(f1 + f2)

    if target_value_0 > target_value_1:
        return 1
    else:
        return 0


def _local_search():
    """
    局部搜索逻辑
    """

    # 2. 根据新的聚类中心计算每个蚂蚁的目标函数

    for i in range(0, ANT_NUM):

        target_value = 0

        for j in range(0, SAMPLE_NUM):

            if ant_array[i][j] == 0:

                # 与分类0的聚类点计算距离
                f1 = math.pow((sample[j][0] - center_array[0][0]), 2)
                f2 = math.pow((sample[j][1] - center_array[0][1]), 2)
                target_value += math.sqrt(f1 + f2)

            else:
                # 与分类1的聚类点计算距离
                f1 = math.pow((sample[j][0] - center_array[1][0]), 2)
                f2 = math.pow((sample[j][1] - center_array[1][1]), 2)
                target_value += math.sqrt(f1 + f2)

                # 保存蚂蚁i当前的目标函数
        ant_target[i] = (i, target_value)

    # 3. 对全部蚂蚁的目标进行排序，选择最优的L只蚂蚁

    ant_target.sort(key=operator.itemgetter(1))  # 对ant进行排序

    # 4. 对这L只蚂蚁进行解的优化
    for i in range(0, L):

        ant_id = ant_target[i][0]

        target_value = 0

        for j in range(0, SAMPLE_NUM):

            # 对于该蚂蚁解集中的每一个样本
            if random.random() < change_jp:
                # 将该样本调整到与当前某个聚类点最近的位置
                t_ant_array[ant_id][j] = _judge_sample(j)

        # 判断是否保留这个临时解
        for j in range(0, SAMPLE_NUM):

            if t_ant_array[ant_id][j] == 0:

                # 与分类0的聚类点计算距离
                f1 = math.pow((sample[j][0] - center_array[0][0]), 2)
                f2 = math.pow((sample[j][1] - center_array[0][1]), 2)
                target_value += math.sqrt(f1 + f2)

            else:
                # 与分类1的聚类点计算距离
                f1 = math.pow((sample[j][0] - center_array[1][0]), 2)
                f2 = math.pow((sample[j][1] - center_array[1][1]), 2)
                target_value += math.sqrt(f1 + f2)

        if target_value < ant_target[i][1]:
            # 更新最优解
            ant_array[ant_id] = t_ant_array[ant_id]


def _update_tau_array():
    """
    更新信息素表
    """
    for i in range(0, SAMPLE_NUM):

        for j in range(0, CLASS_NUM):

            tmp = tao_array[i][j]  # 当前的信息素

            tmp = (1 - change_rho) * tmp  # 处理信息素挥发

            J = 0

            # 处理信息素浓度增加
            for k in range(0, ANT_NUM):

                if ant_array[k][i] == j:
                    f1 = math.pow((sample[i][0] - center_array[j][0]), 2)
                    f2 = math.pow((sample[i][1] - center_array[j][1]), 2)
                    J += math.sqrt(f1 + f2)

            if J != 0:
                tmp += Q / J

                # print(tmp, Q/J)

            tao_array[i][j] = tmp

    # print(np.var(tao_array))


# 全局搜索

def _global_search():
    temp_array = [[0 for col in range(SAMPLE_NUM)] for row in range(ANT_NUM)]
    # 禁忌表，在同一轮迭代中，交换过的蚂蚁，不能再次交换
    taboo = []
    for i in range(int(ANT_NUM / 2)):
        peek_ant1 = 0
        peek_ant2 = 0
        # 交换index1只蚂蚁与index2只蚂蚁特定位置上的值
        while peek_ant1 == peek_ant2 or peek_ant1 in taboo or peek_ant2 in taboo:
            peek_ant1 = random.randint(0, ANT_NUM - 1)
            peek_ant2 = random.randint(0, ANT_NUM - 1)
        # 将已交换的蚂蚁加入禁忌表中
        taboo.append(peek_ant1)
        taboo.append(peek_ant2)
        temp_1 = ant_array[peek_ant1]
        temp_2 = ant_array[peek_ant2]

        for j in range(int(SAMPLE_NUM / 10)):
            index = random.randint(0, SAMPLE_NUM - 1)
            temp_1[index], temp_2[index] = temp_2[index], temp_1[index]
        temp_array[peek_ant1] = temp_1
        temp_array[peek_ant2] = temp_2

    for ant_id in taboo:
        target_value = 0
        # 判断是否保留这个临时解
        for j in range(0, SAMPLE_NUM):

            if t_ant_array[ant_id][j] == 0:

                # 与分类0的聚类点计算距离
                f1 = math.pow((sample[j][0] - center_array[0][0]), 2)
                f2 = math.pow((sample[j][1] - center_array[0][1]), 2)
                target_value += math.sqrt(f1 + f2)

            else:
                # 与分类1的聚类点计算距离
                f1 = math.pow((sample[j][0] - center_array[1][0]), 2)
                f2 = math.pow((sample[j][1] - center_array[1][1]), 2)
                target_value += math.sqrt(f1 + f2)

        if target_value < ant_target[i][1]:
            # 更新最优解
            ant_array[ant_id] = temp_array[ant_id]


from sklearn.cluster import KMeans
import original_test
from sklearn.metrics import precision_score


def run_batch(r):
    _init_test_data(r)

    for i in range(0, ITERATE_NUM):
        _update_ant()
        _global_search()
        _local_search()

        _update_tau_array()

    # 结果集
    pre = ant_array[ant_target[0][0]]
    optimizeAntRes = precision_score(target_classify, pre)

    if (optimizeAntRes > 0.9):
        print("--------------------important!!!!" + "Current OptimizeAntRes:" + str(optimizeAntRes) + "---r:" + str(r))
    else:
        print("Current OptimizeAntRes:" + str(optimizeAntRes) + "---r:" + str(r))


if __name__ == "__main__":

    # r = 1.5
    # while (r<5):
    #     run_batch(r)
    #     r += 0.1

    # i=0
    # while(i<10):
    #     print("=======batch:"+str(i))
    #     run_batch(1.6)
    #     run_batch(2.5)
    #     run_batch(3.1)
    #     run_batch(2.9)
    #     i+=1

    r = 2.5
    # _init_test_data(r)
    change_init_test_data()

    for i in range(0, ITERATE_NUM):
        print("iterate No. {} target {}".format(i, ant_target[0][1]))

        _update_ant()
        # global_optimize()
        _global_search()
        _local_search()

        _update_tau_array()

    # 结果集
    pre = ant_array[ant_target[0][0]]
    optimizeAntRes = precision_score(target_classify, pre)

    plt.figure(figsize=(10, 10), facecolor='w')
    plt.subplot(221)
    plt.title('origin classfication')
    plt.scatter(sample[:, 0], sample[:, 1], c=target_classify, s=20, edgecolors='none')

    plt.subplot(222)
    plt.title('ant classfication')
    plt.scatter(sample[:, 0], sample[:, 1], c=pre, s=20, edgecolors='none')

    plt.plot(center_array[0][0], center_array[0][1], 'ro')
    plt.plot(center_array[1][0], center_array[1][1], 'bo')

    tmp_case, temp_target = ds.make_blobs(100, n_features=2, centers=2, random_state=8)

    model = KMeans(n_clusters=2)
    model.fit(tmp_case)
    km_res = model.predict(sample)
    plt.subplot(223)
    plt.title('KMeans classfication')
    plt.scatter(sample[:, 0], sample[:, 1], c=km_res, s=30, edgecolors='none')

    original_res = original_test.run(sample, target_classify)
    plt.subplot(224)
    plt.title('no update ant')
    plt.scatter(sample[:, 0], sample[:, 1], c=original_res, s=20, edgecolors='none')

    optimizeAntRes = precision_score(target_classify, pre)
    unOptimizeAntRes = precision_score(target_classify, original_res)
    print("优化后准确率：")
    print(optimizeAntRes)
    print("不优化准确率：")
    print(unOptimizeAntRes)

    plt.show()
