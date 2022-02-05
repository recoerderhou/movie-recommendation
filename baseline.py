#  coding -utf-8-
#  二部图，用户看了电影=用户和电影之间有连线

import os
import numpy as np

user = [[0 for col in range(3955)] for row in range(6045)]  # 共6040人，看过电影置1，没看过置0
movie = [0] * 3955  # 存电影名字


#  显示不出来的字符都替换成e
def test_my_file():
    path1 = 'D:\大学\学习\算分\ml-1m'
    filelist = os.listdir(path1)

    for files in filelist:
        Olddir = os.path.join(path1, files)
        filename = os.path.splitext(files)[0]
        filetype = os.path.splitext(files)[1]
        print(Olddir)
        file_test = open(Olddir, 'rb')
        for lines in file_test.readlines():
            #  print(lines)
            strdata = lines.decode().split('::')
            strdata[-1] = strdata[-1].replace('\n', '')
            #  print(strdata)
        file_test.close()


# 用户向量
def get_user_vector():
    path1 = 'D:\大学\学习\算分\ml-1m\\ratings.dat'
    rating_file = open(path1, 'rb')
    for lines in rating_file.readlines():
        rat_data = lines.decode().split('::')
        user[int(rat_data[0])][int(rat_data[1])] = 1  # 看过的打1


# 只需要电影名字
def get_movie_name():
    path1 = 'D:\大学\学习\算分\ml-1m\\movies.dat'
    rating_file = open(path1)
    for lines in rating_file.readlines():
        mov_data = lines.split('::')
        # print(mov_data[1])
        movie[int(mov_data[0])] = mov_data[1]  # 看过的打1


# 余弦相似
def cos_sim(user1, user2):
    vector_a = np.mat(user1)
    # print(vector_a)
    vector_b = np.mat(user2)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim


# 找最相似用户
def select_most_similar(target_user):
    max_sim = -10000
    max_a = 0
    for i in range(0, 6040):
        if i != target_user:
            if cos_sim(user[i], user[target_user]) > max_sim:
                max_a = i
                max_sim = cos_sim(user[i], user[target_user])
    return max_a, max_sim


# 找到最相似用户看过的被推荐用户没看过的电影，挨个推荐
# 如果推荐到了目标用户要看的最后一部电影就算对
def find_movie(max_a, target_user):
    for i in range(0, 3952):
        if user[max_a][i] == 1 and user[target_user][i] == 0:
            # print(movie[i])
            if i == movie_target:
                # print('get')
                return 1
    return 0


get_user_vector()
# print(user)
get_movie_name()
np.random.shuffle(user)
ans = 0
total = 6040
for users in user:
    user_target = user.index(users)
    if sum(users) == 0:
        continue
    movie_target = 0
    for movies in range(len(users) - 1, -1, -1):
        if users[movies] != 0:
            movie_target = movies
            users[movies] = 0
            break
    user_a, max_val = select_most_similar(user_target)
    ans += find_movie(user_a, user_target)

ans = ans / total

print('accuracy: ', ans)
