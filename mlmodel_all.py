import os
from surprise import Reader
from surprise import SVD
from surprise import SVDpp
from surprise import KNNBaseline
from surprise import KNNWithMeans
from surprise import KNNBasic
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise.model_selection import KFold
from surprise.model_selection import GridSearchCV

# Load the movielens-1m
file_path = os.path.expanduser('D:\大学\学习\算分\ml-1m\\ratings.dat')
reader = Reader(line_format='user item rating timestamp', sep='::')
data = Dataset.load_from_file(file_path, reader=reader)

# define a cross-validation iterator
param_grid_svd = {'n_factors': [5, 10, 15, 20], 'biased': [True, False], 'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],
              'reg_all': [0.4, 0.6]}

algo_svd = GridSearchCV(SVD, param_grid_svd)
algo_svd.fit(data)
print('the best score is{}'.format(algo_svd.best_score['rmse']))
print('the best patams is{}'.format(algo_svd.best_params['rmse']))


param_grid_knn = {'k': [10, 20, 30, 40, 50]}
algo_knn = GridSearchCV(KNNBaseline, param_grid_knn)
algo_knn.fit(data)
print('the best score is{}'.format(algo_knn.best_score['rmse']))
print('the best patams is{}'.format(algo_knn.best_params['rmse']))

algo_knn = GridSearchCV(KNNWithMeans, param_grid_knn)
algo_knn.fit(data)
print('the best score is{}'.format(algo_knn.best_score['rmse']))
print('the best patams is{}'.format(algo_knn.best_params['rmse']))

algo_knn = GridSearchCV(KNNBasic, param_grid_knn)
algo_knn.fit(data)
print('the best score is{}'.format(algo_knn.best_score['rmse']))
print('the best patams is{}'.format(algo_knn.best_params['rmse']))


param_grid_svdpp = {'n_factors': [5, 10, 15, 20], 'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],
              'reg_all': [0.4, 0.6]}
algo_svdpp = GridSearchCV(SVDpp, param_grid_svdpp)
algo_svdpp.fit(data)
print('the best score is{}'.format(algo_svdpp.best_score['rmse']))
print('the best patams is{}'.format(algo_svdpp.best_params['rmse']))


