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

SAMPLE_NUM = 150  # 样本数量
FEATURE_NUM = 2  # 每个样本的特征数量
CLASS_NUM = 3  # 分类数量
ANT_NUM = 20  # 蚂蚁数量
ITERATE_NUM = 50  # 迭代次数
NOW_ITER = 1  # 当前迭代轮次
"""
初始化测试样本，sample为样本，target_classify为目标分类结果用于对比算法效果  50
"""
#sample, target_classify = ds.make_blobs(SAMPLE_NUM, n_features=FEATURE_NUM, centers=CLASS_NUM, random_state=40)


data = pd.read_csv('iris_data.csv')
X = data.drop(['target', 'label'], axis=1)
y = data.loc[:, 'label']
# 预处理，将数据规范化
X_norm = StandardScaler().fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_norm)

# iris = ds.load_iris()
# sample = iris.data
# target_classify = iris.target

sample = X_pca
target_classify = data.loc[:, 'label']
"""
信息素矩阵
"""
tao_array = [[[0 for col in range(CLASS_NUM)] for row in range(SAMPLE_NUM)] for ant in range(ANT_NUM)]

"""
蚁群解集
"""
ant_array = [[0 for col in range(SAMPLE_NUM)] for row in range(ANT_NUM)]

t_ant_array = [[0 for col in range(SAMPLE_NUM)] for row in range(ANT_NUM)]  # 存储局部搜索时的临时解

"""
聚类中心点
"""
center_array = [[[0 for col in range(FEATURE_NUM)] for row in range(CLASS_NUM)] for ant in range(ANT_NUM)]

"""
当前轮次蚂蚁的目标函数值，前者是蚂蚁编号、后者是目标函数值
"""
ant_target = [(0, 0) for col in range(ANT_NUM)]  # 生成ANT_NUM个（0，0）

change_q = 0.3  # 更新蚁群时的转换规则参数，表示何种比例直接根据信息素矩阵进行更新
L = int(ANT_NUM / 5)  # 局部搜索的蚂蚁数量
change_jp = 0.3  # 局部搜索时该样本是否变动
change_rho = 0.98  # 挥发参数
Q = 0.1  # 信息素浓度参数


def _init_test_data():
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
    # original_init_center()
    # pick_center_by_density(r)


# 随机选取两个中心点
def original_init_center():
    for i in range(0, CLASS_NUM):
        # 两个属性，i个类
        center_array[i][0] = sample[random.randint(0, SAMPLE_NUM - 1)][0]
        center_array[i][1] = sample[random.randint(0, SAMPLE_NUM - 1)][1]


# 改进后
def getR():
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0
    for i in range(SAMPLE_NUM):
        if (sample[i][0] > sample[max_x][0]):
            max_x = i
        if (sample[i][0] < sample[min_x][0]):
            min_x = i
        if (sample[i][1] > sample[max_y][1]):
            max_y = i
        if (sample[i][1] < sample[max_y][1]):
            min_y = i
    xR = (sample[max_x][0] - sample[min_x][0]) / CLASS_NUM
    yR = (sample[max_y][1] - sample[min_y][1]) / CLASS_NUM
    return max(xR, yR)


def change_init_test_data():
    """
    根据初始聚类中心，建立信息素矩阵
    """
    r = getR()
    # r = 2
    print("r=", r)

    dist = [[0 for col in range(CLASS_NUM)] for row in range(SAMPLE_NUM)]
    for s in range(ANT_NUM):
        r = r * 0.01 * s
        pick_center_by_density(s, r)
        for i in range(SAMPLE_NUM):
            for j in range(CLASS_NUM):
                # ranIndex = random.randint(0, 4)
                dist[i][j] = cal_dis(sample[i], center_array[s][j])
                if (CLASS_NUM * dist[i][j] != 0):
                    tao_array[s][i][j] = 1 / (CLASS_NUM * dist[i][j])
            tmp = _get_best_class_by_tao_value(s, i)
            ant_array[s][i] = tmp

    _init_test_data()


# 改动点1：根据密度选取中心点
def pick_center_by_density(antid, r):
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
    count = 0
    # 备选中心点
    pick_arr_temp = []
    # 最终选取的中心点
    pick_arr = []
    while (count < 3):
        # 最大值
        max_index = findMax(density_arr)

        for k in range(SAMPLE_NUM):
            if density_arr[k] == max_index:
                pick_arr_temp.append(k)
                density_arr[k] = 0
        rand_pick = []
        # 随机选取三个下标
        if (len(pick_arr_temp) >= CLASS_NUM * 2):
            rand_pick = random.sample(pick_arr_temp, CLASS_NUM * 2)
        else:
            rand_pick = random.sample(pick_arr_temp, len(pick_arr_temp))

        for pick in rand_pick:
            if (count == 0):
                center_array[antid][count][0] = sample[pick][0]
                center_array[antid][count][1] = sample[pick][1]
                center_array[antid][count][2] = sample[pick][2]
                pick_arr.append(pick)
                count += 1
            elif count > 0 and count < CLASS_NUM:
                flag = 0
                for oldPick in pick_arr:
                    if (dis_arr[oldPick][pick] < r):
                        flag = 1

                if (flag == 0):
                    center_array[antid][count][0] = sample[pick][0]
                    center_array[antid][count][1] = sample[pick][1]
                    center_array[antid][count][2] = sample[pick][2]
                    pick_arr.append(pick)
                    count += 1


def cal_dis(param, param1):
    x1 = param[0]
    y1 = param[1]
    z1 = param[2]
    x2 = param1[0]
    y2 = param1[1]
    z2 = param[2]
    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2) + math.pow(z1 - z2, 2))


# 找到密度最大的值
def findMax(density_arr):
    index1 = 0

    for i in range(len(density_arr)):
        if density_arr[i] > density_arr[index1]:
            index1 = i
    return density_arr[index1]


def _get_best_class_by_tao_value(antid, sampleid):
    max_value = np.max(tao_array[antid][sampleid])

    for i in range(0, CLASS_NUM):

        if max_value == tao_array[antid][sampleid][i]:
            return i


def random_pick(some_list, probabilities):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability: break
    return item


def _get_best_class_by_tao_probablity(ant, sampleid):
    tarray = np.array(tao_array[ant][sampleid])

    parray = tarray / np.sum(tarray)

    return random_pick([0, 1], parray)


def global_optimize():
    # 产生一个随机数矩阵
    r = np.random.random((ANT_NUM, SAMPLE_NUM))

    for i in range(0, ANT_NUM):
        for j in range(0, SAMPLE_NUM):

            if r[i][j] > change_q:

                tmp_index = _get_best_class_by_tao_value(i, j)

                # 选择该样本中信息素最高的做为分类
                ant_array[i][j] = tmp_index

            else:
                # 计算概率值，根据概率的大小来确定一个选项
                tmp_index = _get_best_class_by_tao_probablity(i, j)

                ant_array[i][j] = tmp_index


# 改进全局搜索

def _global_search():
    temp_array = [[0 for col in range(SAMPLE_NUM)] for row in range(ANT_NUM)]
    # 禁忌表，在同一轮迭代中，交换过的蚂蚁，不能再次交换
    taboo = []
    for i in range(int(ANT_NUM / 3)):
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
    # center_array_temp = center_array
    center_array_temp = [[[0 for col in range(FEATURE_NUM)] for row in range(CLASS_NUM)] for ant in range(ANT_NUM)]

    for i in taboo:

        for j in range(0, CLASS_NUM):
            f_value_feature_0 = []
            f_value_feature_1 = []
            f_value_feature_2 = []
            for k in range(0, SAMPLE_NUM):
                if ant_array[i][k] != j:
                    continue

                f_value_feature_0.append(sample[k][0])  # 簇1
                f_value_feature_1.append(sample[k][1])  # 簇2
                f_value_feature_2.append(sample[k][2])  # 簇3

            if len(f_value_feature_0) > 0:
                center_array_temp[i][j][0] = sum(f_value_feature_0) / len(f_value_feature_0)
            if len(f_value_feature_1) > 0:
                center_array_temp[i][j][1] = sum(f_value_feature_1) / len(f_value_feature_1)
            if len(f_value_feature_2) > 0:
                center_array_temp[i][j][2] = sum(f_value_feature_2) / len(f_value_feature_2)

    t_target = []
    for ant_id in taboo:

        target_value = 0
        # 判断是否保留这个临时解
        for j in range(0, SAMPLE_NUM):

            if temp_array[ant_id][j] == 0:
                # 与分类0的聚类点计算距离
                target_value += cal_dis(sample[j], center_array_temp[ant_id][0])
            if temp_array[ant_id][j] == 1:
                # 与分类1的聚类点计算距离
                target_value += cal_dis(sample[j], center_array_temp[ant_id][1])
            if temp_array[ant_id][j] == 2:
                target_value += cal_dis(sample[j], center_array_temp[ant_id][2])
        t_target.append([ant_id, target_value])

    # 根据目标函数值去做更新
    for target in t_target:
        ant = target[0]
        target = target[1]
        if target < ant_target[ant][1]:
            # 更新最优解
            ant_array[ant] = temp_array[ant]
            center_array[ant] = center_array_temp[ant]
            ant_target[ant] = (ant, target)


def _update_ant_target():
    """
    更新目标函数值
    """
    temp_center_array = [[0 for col in range(FEATURE_NUM)] for row in range(CLASS_NUM)]
    # 1. 确定一个新的聚类中心
    f_value_feature_0 = 0
    f_value_feature_1 = 0
    f_value_feature_2 = 0

    # 目标函数值：属于每个簇的每个样本的每个属性的欧氏距离之和
    # 例如第一簇的所有样本（2，4，7）的三个属性求欧氏距离+第二个簇的所有样本（1，5）的三个属性欧氏距离之和+。。。
    # :[2,2]  7:[3,3]
    # 根号下（1-center0）的平方+

    # 三个属性
    for i in range(0, ANT_NUM):
        target_value = 0
        for j in range(0, SAMPLE_NUM):
            if ant_array[i][j] == 0:
                # 与分类0的聚类点计算距离
                target_value += cal_dis(sample[j], center_array[i][0])

            elif ant_array[i][j] == 1:
                # 与分类1的聚类点计算距离
                target_value += cal_dis(sample[j], center_array[i][1])

            elif ant_array[i][j] == 2:
                # 与分类3的聚类点计算距离
                target_value += cal_dis(sample[j], center_array[i][2])
        if (find_target(i) == 0 or target_value < find_target(i)):
            ant_target[i] = (i, target_value)

    # for j in range(0, ANT_NUM):
    #     f_num_0 = 0
    #     f_num_1 = 0
    #     f_num_2 = 0
    #     target_value = 0
    #     for i in range(CLASS_NUM):
    #         for k in range(0, SAMPLE_NUM):
    #             if ant_array[j][k] == 0:
    #
    #                 f_num_0 += 1
    #                 f_value_feature_0 += sample[k][0]  # 特征1
    #
    #             elif ant_array[j][k] == 1:
    #
    #                 f_num_1 += 1
    #                 f_value_feature_1 += sample[k][1]  # 特征2
    #             elif ant_array[j][k] == 2:
    #
    #                 f_num_2 += 1
    #                 f_value_feature_2 += sample[k][2]  # 特征3
    #
    #         if i == 0 and f_num_0 != 0:
    #             temp_center_array[i][0] = f_value_feature_0 / f_num_0
    #         elif i == 1 and f_num_1 != 0:
    #             temp_center_array[i][1] = f_value_feature_1 / f_num_1
    #         elif i == 2 and f_num_2 != 0:
    #             temp_center_array[i][2] = f_value_feature_2 / f_num_2


def update_ant_center():
    # 中心矩阵：行为簇数，列为属性
    # 求的是属于每个簇的每个样本的各个属性的平均值
    # 例如属于第一个簇的（2，4，7）样本的三个属性的平均值，第二个簇（1，5）三个属性的平均值，。。。

    #
    # for i in range(ANT_NUM):
    #
    #     for j in range(0, CLASS_NUM):
    #         f_value_feature_0 = []
    #         f_value_feature_1 = []
    #         f_value_feature_2 = []
    #         for k in range(0, SAMPLE_NUM):
    #             if ant_array[i][k] != j:
    #                 continue
    #
    #             f_value_feature_0.append(sample[k][0])   # 簇1
    #             f_value_feature_1.append(sample[k][1])   # 簇2
    #             f_value_feature_2.append(sample[k][2])   # 簇3
    #
    #         if len(f_value_feature_0) > 0:
    #             center_array[i][j][0] = sum(f_value_feature_0) / len(f_value_feature_0)
    #         if len(f_value_feature_1) > 0:
    #             center_array[i][j][1] = sum(f_value_feature_1) / len(f_value_feature_1)
    #         if len(f_value_feature_2) > 0:
    #             center_array[i][j][2] = sum(f_value_feature_2) / len(f_value_feature_2)

    center_array_temp = [[[0 for col in range(FEATURE_NUM)] for row in range(CLASS_NUM)] for ant in range(ANT_NUM)]

    for i in range(0, ANT_NUM):

        for j in range(0, CLASS_NUM):
            f_value_feature_0 = []
            f_value_feature_1 = []
            f_value_feature_2 = []
            for k in range(0, SAMPLE_NUM):
                if ant_array[i][k] != j:
                    continue

                f_value_feature_0.append(sample[k][0])  # 簇1
                f_value_feature_1.append(sample[k][1])  # 簇2
                f_value_feature_2.append(sample[k][2])  # 簇3

            if len(f_value_feature_0) > 0:
                center_array_temp[i][j][0] = sum(f_value_feature_0) / len(f_value_feature_0)
            if len(f_value_feature_1) > 0:
                center_array_temp[i][j][1] = sum(f_value_feature_1) / len(f_value_feature_1)
            if len(f_value_feature_2) > 0:
                center_array_temp[i][j][2] = sum(f_value_feature_2) / len(f_value_feature_2)

    for ant_id in range(0, CLASS_NUM):
        target_value = 0
        # 判断是否保留这个临时解
        for j in range(0, SAMPLE_NUM):

            if ant_array[ant_id][j] == 0:
                # 与分类0的聚类点计算距离
                target_value += cal_dis(sample[j], center_array_temp[ant_id][0])
            if ant_array[ant_id][j] == 1:
                # 与分类1的聚类点计算距离
                target_value += cal_dis(sample[j], center_array_temp[ant_id][1])
            if ant_array[ant_id][j] == 2:
                target_value += cal_dis(sample[j], center_array_temp[ant_id][2])

        if target_value < find_target(ant_id):
            # 更新中心矩阵
            center_array[ant_id] = center_array_temp[ant_id]


def _judge_sample(antid, sampleid):
    """
    计算与当前聚类点的举例，判断该sample应所属的归类
    """
    nearly = 0
    minDis = 100
    for i in range(CLASS_NUM):
        dis1 = cal_dis(sample[sampleid], center_array[antid][nearly])
        # f1 = math.pow((sample[sampleid][0] - center_array[antid][nearly][0]), 2)
        # f2 = math.pow((sample[sampleid][1] - center_array[antid][nearly][1]), 2)
        # dis1 = math.sqrt(f1 + f2)

        if (dis1 < minDis):
            minDis = dis1
            nearly = i
    return nearly


#  原局部搜索
def _local_search():
    """
    局部搜索逻辑
    """

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
                t_ant_array[ant_id][j] = _judge_sample(i, j)

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

        if target_value < ant_target[ant_id][1]:
            # 更新最优解
            ant_array[ant_id] = t_ant_array[ant_id]


# 改进后局部搜索
def change_local_search():
    # 4.对所有蚂蚁进行优化
    for i in range(0, ANT_NUM):

        ant_id = ant_target[i][0]

        target_value = 0

        for j in range(0, SAMPLE_NUM):

            # 对于该蚂蚁解集中的每一个样本
            if random.random() < change_jp:
                # 将该样本调整到与当前某个聚类点最近的位置
                t_ant_array[ant_id][j] = _judge_sample(i, j)

        # 判断是否保留这个临时解
        for j in range(0, SAMPLE_NUM):

            if t_ant_array[ant_id][j] == 0:

                # 与分类0的聚类点计算距离
                # f1 = math.pow((sample[j][0] - center_array[i][0][0]), 2)
                # f2 = math.pow((sample[j][1] - center_array[i][0][1]), 2)
                # target_value += math.sqrt(f1 + f2)
                target_value += cal_dis(sample[j], center_array[i][0])
            elif t_ant_array[ant_id][j] == 1:
                # 与分类1的聚类点计算距离
                # f1 = math.pow((sample[j][0] - center_array[i][1][0]), 2)
                # f2 = math.pow((sample[j][1] - center_array[i][1][1]), 2)
                # target_value += math.sqrt(f1 + f2)
                target_value += cal_dis(sample[j], center_array[i][1])
            elif t_ant_array[ant_id][j] == 2:
                # 与分类3的聚类点计算距离
                # f1 = math.pow((sample[j][0] - center_array[i][2][0]), 2)
                # f2 = math.pow((sample[j][1] - center_array[i][2][1]), 2)
                # target_value += math.sqrt(f1 + f2)
                target_value += cal_dis(sample[j], center_array[i][2])
        if target_value < find_target(ant_id):
            # 更新最优解
            ant_array[ant_id] = t_ant_array[ant_id]


# 原更新信息素表
def _update_tau_array():
    """
    更新信息素表
    """
    for n in range(0, ANT_NUM):

        for i in range(0, SAMPLE_NUM):

            for j in range(0, CLASS_NUM):

                tmp = tao_array[n][i][j]  # 当前的信息素

                tmp = (1 - change_rho) * tmp  # 处理信息素挥发

                J = 0

                # 处理信息素浓度增加
                for k in range(0, ANT_NUM):

                    if ant_array[k][i] == j:
                        # f1 = math.pow((sample[i][0] - center_array[n][j][0]), 2)
                        # f2 = math.pow((sample[i][1] - center_array[n][j][1]), 2)
                        # J += math.sqrt(f1 + f2)
                        J += cal_dis(sample[i], center_array[n][j])
                if J != 0:
                    tmp += Q / J

                    # print(tmp, Q/J)

                tao_array[n][i][j] = tmp
            # 根据信息素矩阵更新解字符串
            tmp = _get_best_class_by_tao_value(n, i)
            ant_array[n][i] = tmp
    # print(np.var(tao_array))


# 改进更新信息素表
def change_update_tau_array():
    for n in range(0, ANT_NUM):

        for i in range(0, SAMPLE_NUM):

            for j in range(0, CLASS_NUM):
                tmp = tao_array[n][i][j]  # 当前的信息素
                rou = (ITERATE_NUM - NOW_ITER) / ITERATE_NUM  # 此轮挥发系数
                if rou > 0.02:
                    tmp = rou * tmp  # 处理信息素挥发
                else:
                    tmp = 0.02 * tmp

                # 处理信息素浓度增加

                if ant_array[n][i] == j:
                    tmp = tmp + 1 / find_target[n]

                tao_array[n][i][j] = tmp
        # 根据信息素矩阵更新解字符串
        tmp = _get_best_class_by_tao_value(n, i)
        ant_array[n][i] = tmp


def find_target(ant_id):
    for i in range(0, ANT_NUM):
        if (ant_id == ant_target[i][0]):
            return ant_target[i][1]
    return ant_target[ant_id][1]


from sklearn.cluster import KMeans
import original_test
from sklearn.metrics import precision_score

if __name__ == "__main__":
    change_init_test_data()
    eco_target = []
    for NOW_ITER in range(1, ITERATE_NUM):
        ant_target.sort(key=lambda x: x[1])
        print("iterate No. {} target {}".format(NOW_ITER, ant_target[0][1]))

        _update_ant_target()
        # global_optimize()
        _global_search()
        # _local_search()
        change_local_search()

        update_ant_center()
        # change_update_tau_array()
        _update_tau_array()

        eco_target.append(ant_target[0][1])
    # 结果集
    ant_target.sort(key=lambda x: x[1])
    res = numpy.array(ant_array[ant_target[0][0]])
    optimizeAntRes = precision_score(target_classify, res, average="micro")
    colors1 = '#C0504D'
    colors2 = '#00EEEE'
    colors3 = '#FF6600'

    area1 = np.pi * 2 ** 2  # 半径为2的圆的面积
    area2 = np.pi * 3 ** 2
    area3 = np.pi * 4 ** 2

    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(sample[:, 0][target_classify == 0], sample[:, 1][target_classify == 0],
               sample[:, 2][target_classify == 0], marker='.')
    ax.scatter(sample[:, 0][target_classify == 1], sample[:, 1][target_classify == 1],
               sample[:, 2][target_classify == 1], marker='x')
    ax.scatter(sample[:, 0][target_classify == 2], sample[:, 1][target_classify == 2],
               sample[:, 2][target_classify == 2], marker='*')
    bx = fig.add_subplot(122, projection='3d')
    bx.scatter(sample[:, 0][res == 0], sample[:, 1][res == 0], sample[:, 2][res == 0], marker='.')
    bx.scatter(sample[:, 0][res == 1], sample[:, 1][res == 1], sample[:, 2][res == 1], marker='x')
    bx.scatter(sample[:, 0][res == 2], sample[:, 1][res == 2], sample[:, 2][res == 2], marker='*')

    bx.plot(center_array[0][0][0], center_array[0][0][1], center_array[0][0][2], 'ro')
    bx.plot(center_array[0][1][0], center_array[0][1][1], center_array[0][1][2], 'bo')
    bx.plot(center_array[0][2][0], center_array[0][2][1], center_array[0][2][2], 'yo')
    plt.show()

    plt.figure(figsize=(10, 10), facecolor='w')
    plt.subplot(221)
    plt.title('origin classfication')
    plt.scatter(sample[:, 0][target_classify == 0], sample[:, 1][target_classify == 0], marker='.', s=20)
    plt.scatter(sample[:, 0][target_classify == 1], sample[:, 1][target_classify == 1], marker='x', s=20)
    plt.scatter(sample[:, 0][target_classify == 2], sample[:, 1][target_classify == 2], marker='*', s=20)
    # plt.scatter(sample[:, 0][target_classify == 3], sample[:, 1][target_classify == 3], marker='+', s=20)

    plt.subplot(222)
    plt.title('perfect ant classfication')
    plt.scatter(sample[:, 0][res == 0], sample[:, 1][res == 0], marker='.', s=20)
    plt.scatter(sample[:, 0][res == 1], sample[:, 1][res == 1], marker='x', s=20)
    plt.scatter(sample[:, 0][res == 2], sample[:, 1][res == 2], marker='*', s=20)
    # plt.scatter(sample[:, 0][pre == 3], sample[:, 1][pre == 3], marker='+', s=20)

    plt.plot(center_array[0][0][0], center_array[0][0][1], 'ro')
    plt.plot(center_array[0][1][0], center_array[0][1][1], 'bo')
    plt.plot(center_array[0][2][0], center_array[0][2][1], 'yo')
    # plt.plot(center_array[3][0], center_array[3][1], 'go')
    # print(optimizeAntRes)
    print("优化后准确率：")
    if (optimizeAntRes < 0.5):
        print(1 - optimizeAntRes)
    else:
        print(optimizeAntRes)

    # tmp_case, temp_target = ds.make_blobs(250, n_features=2, centers=2, random_state=30)
    #
    # # kmeans
    # model = KMeans(n_clusters=2)
    # model.fit(tmp_case)
    # km_res = model.predict(sample)
    # plt.subplot(223)
    # plt.title('KMeans classfication')
    # plt.scatter(sample[:, 0], sample[:, 1], c=km_res, s=30, edgecolors='none')
    #
    # # 未优化的蚁群算法
    # original_res = original_test.run(sample, target_classify)
    # plt.subplot(224)
    # plt.title('low ant classfication')
    # plt.scatter(sample[:, 0], sample[:, 1], c=original_res, s=20, edgecolors='none')
    #
    # optimizeAntRes = precision_score(target_classify, pre)
    # unOptimizeAntRes = precision_score(target_classify, original_res)
    # print("优化后准确率：")
    # print(optimizeAntRes)
    # print("不优化准确率：")
    # print(unOptimizeAntRes)
    plt.show()
    plt.figure(figsize=(5, 5), facecolor='w')
    plt.plot(range(ITERATE_NUM - 1), eco_target, linewidth=1, color="orange", marker="o", label="Mean value")
    plt.title("iter and target")

    plt.show()
