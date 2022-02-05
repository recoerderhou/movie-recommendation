# 一阶马尔可夫模型 - 准确率1%
# 通过其他所有人的选择预测这个人的下一步选择
# 矩阵：Aij为所有用户中，看完i紧接着看j的用户占所有用户的比例


import os
import numpy as np
import random as rm

import numpy as np
import random

# 先获得特征向量
global user_time
global user_rate
global rate_latest
global rate_tuple
global markov_matrix

# 使用user_rate 作为输入，rate_latest作为目标输出
user_rate = [[-1] for __ in range(6040)]  # 是用户按时间看的电影的向量
user_time = [[-1] for __ in range(6040)]  # 是用户看电影的时间，排序用
rate_latest = [0 for _ in range(6040)]  # 最后一部电影
rate_tuple = [[-1] for _ in range(6040)]
markov_matrix = [[0 for _ in range(3952)] for _ in range(3952)]


def load_data():
    global user_rate
    global rate_latest
    global user_time

    path3 = 'D:\大学\学习\算分\ml-1m\\ratings.dat'
    rating_file = open(path3, 'rb')
    for lines in rating_file.readlines():
        rat_data = lines.decode().split('::')
        cu = int(rat_data[0]) - 1
        ci = int(rat_data[1]) - 1
        cr = int(rat_data[2])
        ct = int(rat_data[3])  # 时间
        if rate_tuple[cu][0] == -1:
            rate_tuple[cu][0] = (ct, ci)
        else:
            rate_tuple[cu].append((ct, ci))

    # print(rate_tuple[339])
    for i in range(6040):
        rate_tuple[i].sort()
        rate_latest[i] = rate_tuple[i][-1][1]
        for movs in range(len(rate_tuple[i])):
            if user_rate[i][0] == -1:
                user_rate[i][0] = rate_tuple[i][movs][1]
            else:
                user_rate[i].append(rate_tuple[i][movs][1])
    print('finished generating rate features')


def creat_markov_matrix(user):
    global markov_matrix
    global user_rate
    user_rate[user].pop()
    for i in range(6040):
        for j in range(len(user_rate[i]) - 1):
            movie_i = user_rate[i][j]
            movie_j = user_rate[i][j + 1]
            markov_matrix[movie_i][movie_j] += 1 / 3952
    # print(markov_matrix)
    user_rate[user].append(rate_latest[user])
    markov_matrix = np.array(markov_matrix)
    print('finished creating markov_matrix')
    return markov_matrix


def predict(user):
    global markov_matrix
    pred_value = -1
    pred = -1
    latest_movie = user_rate[user][-2]  # 通过倒数第二个预测倒数第一个
    for i in range(3952):
        if markov_matrix[latest_movie][i] > pred_value:
            pred = i
            pred_value = markov_matrix[latest_movie][i]
    return pred

def evaluate_markov():
    ans = 0
    for i in range(6040):
        # print(predict(i), rate_latest[i])
        creat_markov_matrix(i)
        if predict(i) == rate_latest[i]:
            ans += 1
    return ans / 6040


load_data()
print(evaluate_markov())

