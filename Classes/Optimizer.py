import numpy as np 
from sklearn.model_selection import train_test_split
import os, json, yaml, pickle
from sklearn.linear_model import LogisticRegression
# Scikit-Optimisation library
from skopt import gp_minimize
import skopt
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from skopt.utils import use_named_args


class ModelOptimisation:

    def __init__(self, model, num_trials, train_split, validation_split):
        
        with open('settings.yaml') as file:
            Configs = yaml.full_load(file)

        self.dataset_path = Configs['directories']['datasets']
        self.model_path = Configs['directories']['best_model']
        self.num_trials = num_trials
        self.train_split = train_split
        self.validation_split = validation_split
        self.model = model
        self.X = np.loadtxt(f'{self.dataset_path}/X_MinMaxScaler.csv', delimiter = ',')
        self.y = np.loadtxt(f'{self.dataset_path}/y_LabelEncoder.csv', delimiter = ',')

    # Creating the training and testing splits
    def pre_processing(self):
        # Returning the train and test splits
        return train_test_split(self.X, self.y, test_size = 1-self.train_split, random_state = 42, shuffle = True)


    # Function that conducts scikit-learn hyperparameter optimisation - only to be applied when optimising a Non-DL model
    # Before using this function, the user must define the model to be optimised, as well as its hyperparameters
    def Non_DL_optimise(self, search_space: list):
        
        self.train_X, self.test_X, self.train_y, self.test_y = self.pre_processing(self)

        # Defining the evaluator function
        @use_named_args(search_space)

        def evaluate_model(**params):
            # Defining the models and its params
            self.model.set_params(**params)
            # Now fitting the PCA model to the training data
            self.model.fit(self.train_X)
            X_transformed = self.model.transform(self.train_X)
            # Define the model to be used for the classification
            log_reg = LogisticRegression()
            # Define the evaluation procedure
            cv = RepeatedStratifiedKFold(n_splits = 50, n_repeats = 10, random_state = 42)
            # Evaluate the model
            result = cross_val_score(log_reg, X_transformed, self.train_y, cv = cv, n_jobs = -1, scoring = 'accuracy')
            # Return the mean accuracy
            return 1.0 - result.mean()  
        
        # Performing the optimization process
        result = gp_minimize(evaluate_model, search_space, n_calls = self.num_trials, n_random_starts = 10, random_state = 42, verbose = 1)

        return result
    

    # Funcgtion that stores the best model + its hyperparameters
    def store_best_model(self, search_space, result, model_name: str):

        # Getting the best parameters from the optimisation process
        model_parameters, best_parameters = dict(), dict()
        for key, value in zip(search_space, result.x):
            best_parameters[key.name] = str(value)
            model_parameters[key.name] = value

        # Using the best parameters to train the model
        self.model.set_params(**model_parameters)
        self.model.fit(self.train_X)

        print(self.model.get_params())
        # Saving the best parameters dictionary in a file + storing the PCA trained model
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
            with open(f'{self.model_path}/{model_name}.json', 'w') as fp:
                json.dump(best_parameters, fp, indent = 3)
            with open(f'{self.model_path}/{model_name}.pkl', 'wb') as fp:
                pickle.dump(self.model, fp)
        else:
            with open(f'{self.model_path}/{model_name}.json', 'w') as fp:
                json.dump(best_parameters, fp, indent = 3)
            with open(f'{self.model_path}/{model_name}.pkl', 'wb') as fp:
                pickle.dump(self.model, fp)
