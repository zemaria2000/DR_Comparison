
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.decomposition import FastICA
from sklearn.model_selection import train_test_split
import seaborn as sns
import json

# Using some simple classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

# Some results metrics
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix 


# ## Step 1 - Loading data        

X = np.loadtxt('Processed_files/X_StandardScaler.csv', delimiter = ',')
y = np.loadtxt('Processed_Files/y_LabelEncoder.csv', delimiter = ',')
# Creating the train and test splits for data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, shuffle = True)


# ## Step 2 - Optimisation process

from skopt import BayesSearchCV, gp_minimize 
import skopt
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from skopt.utils import use_named_args


# Step 1 - defining the hyperparameters' search space
optimized_parameters = ['n_components', 'algorithm', 'whiten', 'fun', 'max_iter', 'tol', 'whiten_solver']
search_space = []
search_space.append(skopt.space.Integer(2, 30, name = 'n_components'))
search_space.append(skopt.space.Categorical(['parallel', 'deflation'], name = 'algorithm'))
search_space.append(skopt.space.Categorical(['arbitrary-variance', 'unit-variance'], name = 'whiten'))
search_space.append(skopt.space.Categorical(['logcosh', 'exp', 'cube'], name = 'fun'))
search_space.append(skopt.space.Integer(200, 1000, name = 'max_iter'))
search_space.append(skopt.space.Real(1e-5, 1e-3, name = 'tol'))
search_space.append(skopt.space.Categorical(['eigh', 'svd'], name = 'whiten_solver'))

# Step 2 - defining the evaluator
@use_named_args(search_space)

def evaluate_model(**params):

    # Define the model
    model = FastICA()
    model.set_params(**params)

    # Fit the ICA to the training data
    model.fit(X)
    X_ica = model.transform(X)

    # Start the classifier
    log_reg = LogisticRegression()

    # Define the evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits = 50, n_repeats = 10, random_state = 42)

    # Evaluate the model
    result = cross_val_score(log_reg, X_ica, y, cv = cv, n_jobs = -1, scoring = 'accuracy')

    # Return the mean accuracy
    return 1.0 - result.mean()  


# Step 3 - conduct optimisation process
result = gp_minimize(evaluate_model, search_space, n_calls = 5000, n_random_starts = 10, random_state = 42, verbose = 2)

# Analysing the best gotten parameters
result.x
# Storing the parameters in a dictionary
best_parameters = dict()
for key, value in zip(optimized_parameters, result.x):
    best_parameters[key] = str(value)

# Saving the best parameters dictionary in a file
if not os.path.exists('../Models/ICA'):
    os.makedirs('../Models/ICA')
    with open('../Models/ICA/best_parameters.json', 'w') as fp:
        json.dump(best_parameters, fp, indent = 3)
else:
    with open('../Models/ICA/best_parameters.json', 'w') as fp:
        json.dump(best_parameters, fp, indent = 3)

   