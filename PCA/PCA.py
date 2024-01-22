# %%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import json

# Using some simple classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

# Some results metrics
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix 
# %%

# ## Step 1 - Loading data

X = np.loadtxt('../Processed_Files/X_MinMaxScaler.csv', delimiter = ',')
y = np.loadtxt('../Processed_Files/y_LabelEncoder.csv', delimiter = ',')
# Creating a train and test split
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 42, shuffle = True)


# ## Step 2 - Optimisation process of PCA

from skopt import BayesSearchCV, gp_minimize
import skopt
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from skopt.utils import use_named_args

# Step 1 - defining the hyperparameters' search space
optimized_parameters = ['n_components', 'whiten', 'svd_solver', 'n_oversamples', 'power_iteration_normalizer']
search_space = []
search_space.append(skopt.space.Integer(2, 30, name = 'n_components'))
search_space.append(skopt.space.Categorical([True, False], name = 'whiten'))
search_space.append(skopt.space.Categorical(['auto', 'full', 'arpack', 'randomized'], name = 'svd_solver'))
search_space.append(skopt.space.Integer(1, 50, name = 'n_oversamples'))
search_space.append(skopt.space.Categorical(['auto', 'QR', 'LU', 'none'], name = 'power_iteration_normalizer'))

# Step 2 - defining the evaluator
@use_named_args(search_space)

def evaluate_model(**params):

    # Define the model
    model = PCA()
    model.set_params(**params)

    # Now fitting the PCA model to the training data
    model.fit(X)
    X_pca = model.transform(X)

    # Define the model to be used for the classification
    log_reg = LogisticRegression()
    
    # Define the evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits = 50, n_repeats = 10, random_state = 42)

    # Evaluate the model
    result = cross_val_score(log_reg, X_pca, y, cv = cv, n_jobs = -1, scoring = 'accuracy')

    # Return the mean accuracy
    return 1.0 - result.mean()  


# Step 3 - conduct optimisation process
result = gp_minimize(evaluate_model, search_space, n_calls = 5000, n_random_starts = 10, random_state = 42, verbose = 1)

# Analysing the best gotten parameters
result.x
# Storing the parameters in a dictionary
best_parameters = dict()
for key, value in zip(optimized_parameters, result.x):
    best_parameters[key] = str(value)

# Saving the best parameters dictionary in a file
if not os.path.exists('../Models/PCA'):
    os.makedirs('../Models/PCA')
    with open('../Models/PCA/best_parameters.json', 'w') as fp:
        json.dump(best_parameters, fp, indent = 3)
else:
    with open('../Models/PCA/best_parameters.json', 'w') as fp:
        json.dump(best_parameters, fp, indent = 3)

   