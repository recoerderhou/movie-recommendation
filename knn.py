# 协同过滤
# 转化为二分类问题
# user / item 都试一试
# 看没看过 / 评分高低 都试一试
# tf-idf 也试一试
import numpy as np
import random
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

global user
global user_rate
global rate_latest
global item
global class_dict
global gender_dict
global nearest_k

user = [[0, 0, 0, 0] for _ in range(6040)]
user_rate = [[0 for _ in range(3952)] for __ in range(6040)]
rate_latest = [0 for _ in range(6040)]
item = [[0 for _ in range(20)] for _ in range(3952)]
class_dict = {'Action': 0, 'Adventure': 1, 'Animation': 2, 'Children\'s': 3, 'Comedy': 4, 'Crime': 5, 'Documentary': 6,
              'Drama': 7, 'Fantasy': 8, 'Film-Noir': 9, 'Horror': 10, 'Musical': 11, 'Mystery': 12, 'Romance': 13,
              'Sci-Fi': 14, 'Thriller': 15, 'War': 16, 'Western': 17}
gender_dict = {'M': 1, 'F': 0}
nearest_k = 15


def load_data():
    global user
    global user_rate
    global rate_latest
    global item
    global class_dict
    global gender_dict
    print(len(user))
    path1 = 'D:\大学\学习\算分\ml-1m\\users.dat'
    users_file = open(path1, 'rb')
    for lines in users_file.readlines():
        u_data = lines.decode().split('::')
        print(u_data)
        id = int(u_data[0]) - 1
        gender = gender_dict[u_data[1]]
        age = int(u_data[2])
        job = int(u_data[3])
        addr = u_data[4][:-1]
        user[id] = [gender, age, job, addr]
        print(user[id])

    path2 = 'D:\大学\学习\算分\ml-1m\\movies.dat'
    item_file = open(path2, 'rb')
    for lines in item_file.readlines():
        i_data = lines.decode().split('::')
        print(i_data)
        i_feature = [0 for _ in range(20)]
        id = int(i_data[0]) - 1
        i_feature[0] = int(i_data[1][-5:-1])
        i_feature[1] = i_data[1][:-7]
        i_genre = i_data[2][:-1].split('|')
        for genres in i_genre:
            i_feature[2 + class_dict[genres]] = 1
        item[id] = i_feature
        print(item[id])

    path3 = 'D:\大学\学习\算分\ml-1m\\ratings.dat'
    rating_file = open(path3, 'rb')
    latest_time = 0
    for lines in rating_file.readlines():
        rat_data = lines.decode().split('::')
        cu = int(rat_data[0]) - 1
        ci = int(rat_data[1]) - 1
        cr = int(rat_data[2])
        ct = int(rat_data[3])
        user_rate[cu][ci] = cr  # 看过的打1 or 看过的打分
        if ct > latest_time:
            rate_latest[cu] = ci
            latest_time = ct


# 构造数据集
def prepare_test():
    global user
    global user_rate
    global rate_latest
    global item
    total_set = list(zip(user_rate, rate_latest))
    random.shuffle(total_set)
    total_data, total_label = zip(*total_set)
    total_data = list(total_data)
    total_label = list(total_label)
    test_size = 604
    test_data = total_data[:test_size]
    test_label = total_label[:test_size]
    train_data = total_data[test_size:]
    for datas in range(604):
        test_data[datas][test_label[datas]] = 0  # 将最后一个评分的项抹除
    print('test_prepare finished')
    return test_data, test_label, train_data


def similarity_calc(user1, user2):
    a = [user1, user2]
    ans = cosine_similarity(a)
    return ans[0][1]


def softmax(x):
    x_exp = np.exp(x)
    #如果是列向量，则axis=0
    x_sum = np.sum(x_exp, keepdims=True)
    s = x_exp / x_sum
    return s


def user_cf():
    acc = 0
    total = 604
    # test data/label & train data/label
    td, tl, rd = prepare_test()
    k_store = [-1 for _ in range(15)]  # 存储最相近的前15个用户的id
    k_max = [-100 for _ in range(15)]  # 存储最相近的前十五个用户的相似度
    for users in td:
        print('testing_user ', td.index(users))
        p_movies = []  # 相似度前15的用户看过的电影中目标用户没看过的
        p_rated = []  # 相似度前15的用户看过↑的人数
        for trainer in rd:
            sim = similarity_calc(users, trainer)
            # print('similarity: ', sim)
            for i in range(15):
                if sim > k_max[i]:
                    k_max[i] = sim
                    k_store[i] = rd.index(trainer)
                    break
        # k_ans = softmax(k_max)  # 得到权重
        print('now similarest is: ', k_store)
        for i in range(3952):
            if users[i]:
                continue
            # print('not watched')
            num = 0
            for tops in k_store:
                num += rd[tops][i]
            if num:
                # print('yes')
                p_movies.append(i)
                p_rated.append(num)

        for movie in p_movies:
            if movie == tl[td.index(users)]:
                print('found')
                acc += 1
                break
        k_store = [-1 for _ in range(15)]  # 每个用户都重新算
        k_max = [-100 for _ in range(15)]

    acc = acc / total

    print('current accuracy is :', acc)


load_data()
user_cf()
