# Script that creates all the search parameters for Non-DL models
# Non-DL models so far - PCA, ICA

import skopt
import pickle
import os

# PCA
ss_pca = []
ss_pca.append(skopt.space.Integer(2, 30, name = 'n_components'))
ss_pca.append(skopt.space.Categorical([True, False], name = 'whiten'))
ss_pca.append(skopt.space.Categorical(['auto', 'full', 'arpack', 'randomized'], name = 'svd_solver'))
ss_pca.append(skopt.space.Integer(1, 50, name = 'n_oversamples'))
ss_pca.append(skopt.space.Categorical(['auto', 'QR', 'LU', 'none'], name = 'power_iteration_normalizer'))

# FastICA
ss_ica = []
ss_ica.append(skopt.space.Integer(2, 30, name = 'n_components'))
ss_ica.append(skopt.space.Categorical(['parallel', 'deflation'], name = 'algorithm'))
ss_ica.append(skopt.space.Categorical(['arbitrary-variance', 'unit-variance'], name = 'whiten'))
ss_ica.append(skopt.space.Categorical(['logcosh', 'exp', 'cube'], name = 'fun'))
ss_ica.append(skopt.space.Integer(200, 1000, name = 'max_iter'))
ss_ica.append(skopt.space.Real(1e-5, 1e-3, name = 'tol'))
ss_ica.append(skopt.space.Categorical(['eigh', 'svd'], name = 'whiten_solver'))


# Saving the best parameters dictionary in a file + storing the PCA trained model
if not os.path.exists('./SearchSpace'):
    os.makedirs('./SearchSpace')

with open(f'./SearchSpace/ss_ica.txt', 'wb') as fp:
    pickle.dump(ss_ica, fp)
with open(f'./SearchSpace/ss_pca.txt', 'wb') as fp:
    pickle.dump(ss_pca, fp)

