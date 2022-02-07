import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
mh = ["Movie_id","Title","Action","Adventure","Animation","Children's","Comedy","Crime","Documentary",
      "Drama","Fantasy","Film-noir","Horror","Musical","Mystery","Romance","Sci-Fi",
      "Thriller","War","Western"]
movies = pd.read_table('movieone.csv', sep=',', header=None, names=mh,index_col = 0, engine = 'python')
#print(movies)
total = np.zeros([18])
#print(d)
a = movies.sum(axis=0)
for i in range(18):
    total[i] = a[i+1]
plt.gcf().subplots_adjust(bottom=0.3)
plt.figure(dpi=60)
plt.figure('Movie Genres',facecolor='pink')
plt.title("Movie Genres",fontsize=16)
plt.xlabel('Genres',fontsize=14)
plt.ylabel('Sum',fontsize=14)
plt.ylim(0,1800)
x=np.arange(18)
plt.bar(x,total,0.4,align = 'center')
#plt.xticks(np.arange(18),[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
plt.xticks(np.arange(18),mh[2:],size='small',rotation='vertical',fontsize=13)
for a,b in zip(x,total):
    plt.text(a,b+0.05, '%.0f' %b, ha='center',va='bottom',fontsize=10)
plt.show()
