
#  coding:-utf-8-
#  svd++

import numpy as np

global user
user = np.zeros((6040, 3952))  # 用户x电影的评分矩阵
# user = np.array(user)
global rated_items
rated_items = [0] * 6040
rated_items = np.array(rated_items)  # 评过分的电影数目
global rated_ids
rated_ids = [[-1] for _ in range(6040)]  # 每个用户评过分的电影集合
global global_mean
global_mean = 0.0 # 所有记录的评分的全局平均数
global total_num
total_num = 0 # 网站总共的评分数


def creat_matrix():
    global global_mean
    global total_num
    global rated_items
    path1 = 'D:\大学\学习\算分\ml-1m\\ratings.dat'
    rating_file = open(path1, 'rb')
    for lines in rating_file.readlines():
        rat_data = lines.decode().split('::')
        # print(int(rat_data[2]))
        cu = int(rat_data[0]) - 1
        ci = int(rat_data[1]) - 1
        cr = int(rat_data[2])
        user[cu][ci] = cr  # 看过的打1
        # print(user[int(rat_data[0])][int(rat_data[1])])
        rate = int(rat_data[2])
        global_mean = rate + global_mean
        total_num += 1
        rated_items[cu] += 1
        if len(rated_ids[cu]) == 1 and rated_ids[cu][0] == -1:
            rated_ids[cu] = [ci]
        else:
            rated_ids[cu].append(ci)


creat_matrix()
print(user)
print(len(rated_ids[20]))
print(len(rated_ids[190]))
print(rated_items[1:6040])
global_mean = global_mean / total_num
print(global_mean)


def svd(feature, steps=500, gama=0.02, lamda=0.3):
    global global_mean
    global rated_items
    global rated_ids
    global user

    slowRate = 0.99

    print('?')
    bu = np.random.rand(user.shape[0])
    bi = np.random.rand(user.shape[1])
    p = np.random.rand(user.shape[0], feature)
    q = np.random.rand(user.shape[1], feature)
    y = np.random.rand(user.shape[1], feature)

    print(bu)
    print(p)
    print(y)

    # print(user_feature)
    # print(item_feature)

    for step in range(steps):
        print("processing epoch {}".format(step))
        n = 0
        for u in range(user.shape[0]):
            for i in range(user.shape[1]):
                if user[u, i]:
                    print('user rate is : ',user[u, i])
                    Nu = rated_items[u], sqrt_Nu = np.sqrt(Nu)
                    y_u = np.sum(y[rated_ids[u]], axis=0)
                    u_impl_prf = y_u / sqrt_Nu
                    rp = global_mean + bu[u] + bi[i] + np.dot(q[i], (p[u] + u_impl_prf).T)
                    print('rp is: ', rp)
                    if rp > 5: rp = 5
                    if rp < 1: rp = 1
                    eui = user[u, i] - rp
                    print('eui is : ', eui)
                    n += 1
                    bu[u] += gama * (eui - lamda * bu[u])
                    bi[i] += gama * (eui - lamda * bi[i])
                    p[u] += gama * (eui * q[i] - lamda * p[u])
                    q[i] += gama * (eui * p[u] - lamda * q[i])
                    for j in rated_ids[u]: y[j] += gama * (eui * q[j] / sqrt_Nu - lamda * y[j])
        print('step: %d ' % ((step + 1)))
        gama *= slowRate
    return p, q, bu, bi, y


global p, q, bu, bi, y


def predict(users, item):
    global global_mean
    global rated_items
    global user
    global rated_ids
    print(user)
    pp, q, bu, bi, y = svd(15)
    Nu = rated_items[users]
    sqrt_Nu = np.sqrt(Nu)
    y_u = np.sum(y[rated_ids[users]], axis=0) / sqrt_Nu
    predict = global_mean + bu[users] + bi[item] + np.dot(q[item], p[users] + y_u)
    return predict


users = 10
item = 20
predict(users, item)
