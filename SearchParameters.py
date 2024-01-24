# Script that creates all the search parameters for Non-DL models
# Non-DL models so far - PCA, ICA

import skopt
import pickle
import os, yaml
import numpy as np

# Loading the config file
with open('settings.yaml') as file:
    Configs = yaml.full_load(file)

# Just loading the X and y data
data_path = Configs['directories']['datasets']
X = np.loadtxt(f'{data_path}/X_MinMaxScaler.csv', delimiter = ',')
y = np.loadtxt(f'{data_path}/y_LabelEncoder.csv', delimiter = ',')

# PCA
ss_pca = []
ss_pca.append(skopt.space.Integer(2, X.shape[1], name = 'n_components'))
ss_pca.append(skopt.space.Categorical([True, False], name = 'whiten'))
ss_pca.append(skopt.space.Categorical(['auto', 'full', 'arpack', 'randomized'], name = 'svd_solver'))
ss_pca.append(skopt.space.Integer(1, 50, name = 'n_oversamples'))
ss_pca.append(skopt.space.Categorical(['auto', 'QR', 'LU', 'none'], name = 'power_iteration_normalizer'))

# FastICA
ss_ica = []
ss_ica.append(skopt.space.Integer(2, X.shape[1], name = 'n_components'))
ss_ica.append(skopt.space.Categorical(['parallel', 'deflation'], name = 'algorithm'))
ss_ica.append(skopt.space.Categorical(['arbitrary-variance', 'unit-variance'], name = 'whiten'))
ss_ica.append(skopt.space.Categorical(['logcosh', 'exp', 'cube'], name = 'fun'))
# ss_ica.append(skopt.space.Integer(200, 5000, name = 'max_iter'))
ss_ica.append(skopt.space.Real(1e-5, 1e-3, name = 'tol'))
ss_ica.append(skopt.space.Categorical(['eigh', 'svd'], name = 'whiten_solver'))

# SVD
ss_svd = []
ss_svd.append(skopt.space.Integer(2, X.shape[1], name = 'n_components'))
ss_svd.append(skopt.space.Categorical(['arpack', 'randomized'], name = 'algorithm'))
ss_svd.append(skopt.space.Categorical(['auto', 'QR', 'LU', 'none'], name = 'power_iteration_normalizer'))

# LDA
ss_lda = []
ss_lda.append(skopt.space.Integer(2, X.shape[1], name = 'n_components'))
ss_lda.append(skopt.space.Categorical(['svd', 'lsqr', 'eigen'], name = 'solver'))
ss_lda.append(skopt.space.Real(0, 1, name = 'shrinkage'))

# RF (feature selection)
ss_rf = []
ss_rf.append(skopt.space.Integer(15, 100, name = 'n_estimators'))
ss_rf.append(skopt.space.Categorical(['gini', 'entropy', 'log_loss'], name = 'criterion'))
ss_rf.append(skopt.space.Categorical(['sqrt', 'log2', None], name = 'max_features'))
ss_rf.append(skopt.space.Categorical([True, False], name = 'bootstrap'))


# Saving the best parameters dictionary in a file + storing the PCA trained model
if not os.path.exists('./SearchSpace'):
    os.makedirs('./SearchSpace')

with open(f'./SearchSpace/ss_ica.txt', 'wb') as fp:
    pickle.dump(ss_ica, fp)
with open(f'./SearchSpace/ss_pca.txt', 'wb') as fp:
    pickle.dump(ss_pca, fp)
with open(f'./SearchSpace/ss_svd.txt', 'wb') as fp:
    pickle.dump(ss_svd, fp)
with open(f'./SearchSpace/ss_lda.txt', 'wb') as fp:
    pickle.dump(ss_lda, fp)
with open(f'./SearchSpace/ss_rf.txt', 'wb') as fp:
    pickle.dump(ss_rf, fp)