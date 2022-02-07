import pandas as pd
import numpy as np
movief = pd.read_excel('mf_16.xlsx')
#movief = pd.read_excel('mf_64.xlsx')
userf = pd.read_excel('uf_16.xlsx')
#userf = pd.read_excel('uf_64.xlsx')
movies = movief.values
users = userf.values

movie = []
user = []
for i in range(len(movies)):
    a=movies[i][1].split(',')
    tpl = []
    for j in range(len(a)):
        tpl.append(float(a[j]))
    movie.append(tpl)
for i in range(len(users)):
    a=users[i][1].split(',')
    tpl = []
    for j in range(len(a)):
        tpl.append(float(a[j]))
    user.append(tpl)

import torch
import random
moviet = torch.from_numpy(np.array(movie))
usert = torch.from_numpy(np.array(user))
def int_random(k):
    ls = []
    while len(ls)<k:
        a=random.randint(0,19)
        if(a not in ls):
            ls.append(a)
    return ls

from math import sqrt
def multipl(a,b):
    sumofab = 0.0
    for i in range(len(a)):
        temp = a[i] * b[i]
        sumofab += temp
    return sumofab
def corrcoef(x,y):
    n = len (x)
    sum1 = sum (x)
    sum2 = sum (y)
    sumofxy = multipl(x,y)
    sumofx2 = sum([pow(i,2) for i in x])
    sumofy2 = sum([pow(j,2) for j in y])
    num = sumofxy - (float(sum1)*float(sum2)/n)
    den = sqrt((sumofx2 - float(sum1**2)/n)*(sumofy2 - float(sum2**2)/n))
    return num/den


'''实现三种推荐方式：topK默认为20
1：输入电影ID，推荐相似度最高的5部电影，topK选5
2：输入用户ID，推荐最有可能观看的5部电影，topK选5
3：输入用户ID，推荐相似度最高的5个人分别最有可能看的5部电影，topK选5'''

# return the indexes of movie
# dist function: Euclid distance
def recommend_by_movie(movieIndex,topK=20):
    a = moviet[movieIndex]
    ls1 = []
    for i in range(len(moviet)):
        if (i== movieIndex):
            continue
        ls1.append(torch.dist(moviet[i],a,p=2))
    am = torch.from_numpy(np.array(ls1))
    values, indices = torch.topk(am,topK,largest=False)
    ran = int_random(5)
    recoml = []
    for i in range(len(ran)):
        recoml.append(indices[ran[i]].item())
    return recoml
#print(recommend_by_movie(20))


# return the indexes of movie
# dist function: Pearson correlation
def recommend_by_user_self(userIndex,topK=20):
    a = usert[userIndex]
    ls1 = []
    for i in range(len(moviet)):
        ls1.append(corrcoef(a,moviet[i]))
    am = torch.from_numpy(np.array(ls1))
    values, indices = torch.topk(am,topK,largest=True)
    ran = int_random(5)
    recoml = []
    for i in range(len(ran)):
        recoml.append(indices[ran[i]].item())
    return recoml
#print(recommend_by_user_self(1010))


# return the list of five users and their movies
# dist fumction; Hammming distance
def recommend_by_user_other(userIndex,topK=20):
    a = usert[userIndex]
    ls1 = []
    for i in range(len(usert)):
        if(i==userIndex):
            continue
        ls1.append(torch.dist(usert[i],a,p=1))
    au=torch.from_numpy(np.array(ls1))
    values,indices = torch.topk(au,topK,largest=False)
    ran = int_random(5)
    simuser = []
    for i in range(len(ran)):
        simuser.append(indices[ran[i]].item())
    recoml=[]
    for i in range(len(simuser)):       #userIndex
        recoml.append(recommend_by_user_self(simuser[i]))
    return simuser,recoml
#print(recommend_by_user_other(20))



import easygui as eg

movieindex = pd.read_table('movies.dat', sep='::', names = ['ID','name','genres'], encoding = 'latin1', engine='python')
userindex = pd.read_table('users.dat',sep='::',names=['ID','gender','age','occupation','zipcode'],engine='python')

occu = ['other','academic/eductor','artist','clerical/admin','college/grad student',
     'customer service','doctor/health care','executive/managerial','farmer','homemaker',
     'K-12 student','lawyer','programmer','retired','sales/marketing','scientist',
      'self-emplyed','technician/engineer','tradesman/craftsman','unemplyed','writer']

def getMindex(num):
    a = movieindex[(movieindex['ID']==int(num))].index.to_list()
    return int(a[0])
def getUindex(num):
    a = userindex[(userindex['ID']==int(num))].index.to_list()
    return int(a[0])

def Guide():
    eg.msgbox(msg = 'Choose your reommendation methods first',
                      title  = 'Guide', ok_button = 'Got it!')

def Choice():
    ret = eg.choicebox(msg = 'Choose your recommendation methods:',
                   title = 'Methods for Recommendation',
                   choices = (['Methods 1','Methods 2','Methods 3']))
    return ret

def rec_1():
    shuru = eg.enterbox(msg='Input the movieID',
                        title = 'Recommendation by similar movie')
    string = "The movie ID is "+str(shuru)+" and its most similar movies are:\n"
    shuru = getMindex(shuru)
    ls = recommend_by_movie(int(shuru))
    for i in range(len(ls)):
        a = movieindex.iloc[ls[i]].values.tolist()
        string += ('Movie '+str(i+1)+':\t ID:'+str(a[0])+',\t Title:'+a[1]+',\t Genres:'+a[2]+'\n')
    eg.msgbox(msg=string,title='Your recommendation',ok_button='Got it!')
    
    
def rec_2():
    shuru = eg.enterbox(msg='Input the userID',
                        title = 'Recommendation by yourself')
    string = "Your userID: "+str(shuru)+'\nYour most likely loving movies are:\n'
    shuru = getUindex(shuru)
    ls = recommend_by_user_self(int(shuru))
    for i in range(len(ls)):
        a = movieindex.iloc[ls[i]].values.tolist()
        string += ('Movie '+str(i+1)+':\t ID:'+str(a[0])+',\t Title:'+a[1]+',\t Genres:'+a[2]+'\n')
    eg.msgbox(msg=string,title='Your recommendation',ok_button='Got it!')
        
def rec_3():
    shuru = eg.enterbox(msg='Input the userID',
                        title = 'Recommendation by other users')
    string = "Your userID: "+str(shuru)+'\n'
    shuru = getUindex(shuru)
    ls = recommend_by_user_other(int(shuru))        #index
    string += "Your fimilar users and their most likely loving movies are:\n\n"
    for i in range(len(ls[0])):
        a = userindex.iloc[ls[0][i]].values.tolist()
        string+=("User "+str(i+1)+':\t ID:'+str(a[0])+',\t Gender:')
        if(a[1]=='F'):
            string+='Female'
        else:
            string+='Male'
        string+=',\t Age:'
        if(a[2]==1):
            string+='Under 18'
        elif(a[2]==18):
            string+='18-24'
        elif(a[2]==25):
            string+='25-34'
        elif(a[2]==35):
            string+='35-44'
        elif(a[2]==45):
            string+='45-49'
        elif(a[2]==50):
            string+='50-55'
        else:
            string+='56 and beyond'
        string+=(',\t Occupation:'+occu[a[3]])
        string+=(',\t Zipcode:'+str(a[4])+'\n')
        if(a[1]=='F'):
            string+=("Her favorite movies are:\n")
        else:
            string+=("His favorite movies are:\n")
        
        mylist = ls[1][i]
        #print(mylist)
        for j in range(len(mylist)):
            a2 = movieindex.iloc[mylist[j]].values.tolist()
            string += ('Movie '+str(j+1)+':\t ID:'+str(a2[0])+',\t Title:'+a2[1]+',\t Genres:'+a2[2]+'\n')
        string+='\n'
    eg.msgbox(msg=string,title='Your recommendation',ok_button='Got it!')
            
    
Guide()
while(True):
    choice = Choice()
    if(choice == 'Methods 1'):
        rec_1()
    elif(choice == 'Methods 2'):
        rec_2()
    elif(choice == 'Methods 3'):
        rec_3()
    else:
        Guide()
    
