# Script that creates all the search parameters for Non-DL models
# Non-DL models so far - PCA, ICA

import skopt
import pickle
import os, yaml
import numpy as np

# Loading the config file
with open('settings.yaml') as file:
    Configs = yaml.full_load(file)

search_space_path = Configs['directories']['search_space']

# Just loading the X and y data
data_path = Configs['directories']['datasets']
X = np.loadtxt(f'{data_path}/X_MinMaxScaler.csv', delimiter = ',')
y = np.loadtxt(f'{data_path}/y_LabelEncoder.csv', delimiter = ',')

# PCA
ss_pca = []
ss_pca.append(skopt.space.Integer(2, X.shape[1]-1, name = 'n_components'))
ss_pca.append(skopt.space.Categorical([True, False], name = 'whiten'))
ss_pca.append(skopt.space.Categorical(['auto', 'full', 'arpack', 'randomized'], name = 'svd_solver'))
ss_pca.append(skopt.space.Integer(1, 50, name = 'n_oversamples'))
ss_pca.append(skopt.space.Categorical(['auto', 'QR', 'LU', 'none'], name = 'power_iteration_normalizer'))

# FastICA
ss_ica = []
ss_ica.append(skopt.space.Integer(2, X.shape[1]-1, name = 'n_components'))
ss_ica.append(skopt.space.Categorical(['parallel', 'deflation'], name = 'algorithm'))
ss_ica.append(skopt.space.Categorical(['arbitrary-variance', 'unit-variance'], name = 'whiten'))
ss_ica.append(skopt.space.Categorical(['logcosh', 'exp', 'cube'], name = 'fun'))
# ss_ica.append(skopt.space.Integer(200, 5000, name = 'max_iter'))
ss_ica.append(skopt.space.Real(1e-5, 1e-3, name = 'tol'))
ss_ica.append(skopt.space.Categorical(['eigh', 'svd'], name = 'whiten_solver'))

# SVD
ss_svd = []
ss_svd.append(skopt.space.Integer(2, X.shape[1]-1, name = 'n_components'))
ss_svd.append(skopt.space.Categorical(['arpack', 'randomized'], name = 'algorithm'))
ss_svd.append(skopt.space.Categorical(['auto', 'OR', 'LU', 'none'], name = 'power_iteration_normalizer'))

# LDA
ss_lda = []
ss_lda.append(skopt.space.Integer(2, len(np.unique(y))-1, name = 'n_components'))
# ss_lda.append(skopt.space.Categorical(['svd', 'eigen'], name = 'solver'))
# ss_lda.append(skopt.space.Real(0, 1, name = 'shrinkage'))

# RF (feature selection)
ss_rf = []
ss_rf.append(skopt.space.Integer(15, 100, name = 'n_estimators'))
ss_rf.append(skopt.space.Categorical(['gini', 'entropy', 'log_loss'], name = 'criterion'))
ss_rf.append(skopt.space.Categorical(['sqrt', 'log2', None], name = 'max_features'))
ss_rf.append(skopt.space.Categorical([True, False], name = 'bootstrap'))

# NMF (Non-negative Matrix Factorization)
ss_nmf = []
ss_nmf.append(skopt.space.Integer(2, X.shape[1]-1, name = 'n_components'))
ss_nmf.append(skopt.space.Categorical(['random', 'nndsvd', 'nndsvda', 'nndsvdar'], name = 'init'))
# ss_nmf.append(skopt.space.Categorical(['cd', 'mu'], name = 'solver'))
# ss_nmf.append(skopt.space.Categorical(['frobenius', 'kullback-leibler', 'itakura-saito'], name = 'beta_loss'))
# ss_nmf.append(skopt.space.Categorical(['kullback-leibler', 'itakura-saito'], name = 'beta_loss'))

# RFEV (feature selection, Recursive Feature Elimination with Cross-Validation)
ss_rfe = []
ss_rfe.append(skopt.space.Integer(2, X.shape[1]-1, name = 'min_features_to_select'))
ss_rfe.append(skopt.space.Real(0.01, 0.2, name = 'step'))
ss_rfe.append(skopt.space.Integer(2, 20, name = 'cv'))
# NOTE: I need to use an estimator for the rfev, and the method needs to contain a .features_importances_ attribute

# Saving the best parameters dictionary in a file + storing the PCA trained model
if not os.path.exists(f'{search_space_path}'):
    os.makedirs(f'{search_space_path}')

with open(f'{search_space_path}/ss_ica.txt', 'wb') as fp:
    pickle.dump(ss_ica, fp)
with open(f'{search_space_path}/ss_pca.txt', 'wb') as fp:
    pickle.dump(ss_pca, fp)
with open(f'{search_space_path}/ss_svd.txt', 'wb') as fp:
    pickle.dump(ss_svd, fp)
with open(f'{search_space_path}/ss_lda.txt', 'wb') as fp:
    pickle.dump(ss_lda, fp)
with open(f'{search_space_path}/ss_rf.txt', 'wb') as fp:
    pickle.dump(ss_rf, fp)
with open(f'{search_space_path}/ss_nmf.txt', 'wb') as fp:
    pickle.dump(ss_nmf, fp)
with open(f'{search_space_path}/ss_rfe.txt', 'wb') as fp:
    pickle.dump(ss_rfe, fp)