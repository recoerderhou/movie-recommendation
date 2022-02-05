# 最基本的svd

import numpy as np

user = [[0 for col in range(3955)] for row in range(6045)]  # 共6040人，看过电影置1，没看过置0


def creat_matrix():
    path1 = 'D:\大学\学习\算分\ml-1m\\ratings.dat'
    rating_file = open(path1, 'rb')
    for lines in rating_file.readlines():
        rat_data = lines.decode().split('::')
        # print(int(rat_data[2]))
        user[int(rat_data[0])][int(rat_data[1])] = int(rat_data[2])  # 看过的打1
        # print(user[int(rat_data[0])][int(rat_data[1])])


creat_matrix()
print(user)


def svd(mat, feature, steps=500, gama=0.02, lamda=0.3):
    slowRate = 0.99
    preRmse = 1000000000.0
    nowRmse = 0.0

    user_feature = np.matrix(np.random.rand(mat.shape[0], feature))
    item_feature = np.matrix(np.random.rand(mat.shape[1], feature))
    print(user_feature)
    print(item_feature)

    for step in range(steps):
        rmse = 0.0
        n = 0
        for u in range(mat.shape[0]):
            if u % 500 == 0:
                print('testing user ', u)
            for i in range(mat.shape[1]):
                if mat[u, i]:
                    pui = float(np.dot(user_feature[u, :], item_feature[i, :].T))
                    eui = mat[u, i] - pui
                    # print('eui is ', eui)
                    rmse += pow(eui, 2)
                    n += 1
                    for k in range(feature):
                        user_feature[u, k] += gama * (eui * item_feature[i, k] - lamda * user_feature[u, k])
                        item_feature[i, k] += gama * (
                                    eui * user_feature[u, k] - lamda * item_feature[i, k])
        nowRmse = np.sqrt(rmse * 1.0 / n)
        print('step: %d      Rmse: %s' % ((step + 1), nowRmse))
        if (nowRmse < preRmse):
            preRmse = nowRmse
        else: break
        gama *= slowRate
        step += 1
    return user_feature, item_feature


ufe, ife = svd(np.array(user), 15)
print(ufe)
print(ife)
