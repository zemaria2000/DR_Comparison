from Classes import Testing
import yaml, sys
import pickle, json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression

args = sys.argv

# --------------------------------------------------------------------------------
# ------------------- GETTING/DEFINING SOME IMPORTANT SETTINGS -------------------
# --------------------------------------------------------------------------------

# Importing some settings
with open('settings.yaml') as file:
    Configs = yaml.full_load(file)

models_list, DL_models_list = Configs['models']['available_models'], Configs['models']['DL_models']
params_path = Configs['directories']['parameters']

# Defining some important variables at the beginning of this script
n_tests = 5
n_components_list = [2, 5, 10, 15, 20, 25, 30, 35, 40]
classifier = LogisticRegression()
ae_dims_list = [48, 32, 16, 8, 4, 2]



# --------------------------------------------------------------------------------
# ---------------------- DEFINING SOME IMPORTANT METHODS -------------------------
# --------------------------------------------------------------------------------


# Defining some important functions to conduct the tests
def test_default_models(default_model, n_components, n_tests, classifier):

    # Create an empty dataframe to store the results
    results_df = pd.DataFrame()

    if model_name != 'AE':

        for n_components in n_components_list:

            # Instantiating the best results series for each model
            best_test = pd.Series()

            for i in range(n_tests):

                # Fit the DR technique 
                trained_model = tester.fit_DR_technique(default_model, n_components = n_components)
                # Generate the reduced datasets
                train_X_reduced, test_X_reduced, train_y, test_y = tester.generate_reduced_datasets_NonDL(trained_model, n_components)
                # Fit the classifier
                classifier.fit(train_X_reduced, train_y)
                # Make the predictions
                y_pred = classifier.predict(test_X_reduced)
                # Evaluate the model
                metrics = tester.get_metrics(test_y, y_pred, n_components)
                if best_test.empty:
                    best_test = metrics
                if best_test['F1'] < metrics['F1']:
                    best_test = metrics
                else:
                    pass
            # Store the best result for each test
            results_df = results_df.append(best_test, ignore_index = True)
    
    else:
            
        latent_space_dims = ae_dims_list[::-1]
        latent_space_dims.pop()
        
        for n_components in latent_space_dims:

            # Instantiating the best results series for each model
            best_test = pd.Series()
            default_model = tester.build_ae(ae_dims_list)

            for i in range(n_tests):

                # Fit the DR technique 
                trained_model = tester.fit_DR_technique(default_model, n_components = n_components)
                # Generate the reduced datasets
                train_X_reduced, test_X_reduced, train_y, test_y = tester.generate_reduced_datasets_DL(trained_model)
                # Fit the classifier
                classifier.fit(train_X_reduced, train_y)
                # Make the predictions
                y_pred = classifier.predict(test_X_reduced)
                # Evaluate the model
                metrics = tester.get_metrics(test_y, y_pred, n_components)
                if best_test.empty:
                    best_test = metrics
                if best_test['F1'] < metrics['F1']:
                    best_test = metrics
                else:
                    pass
            # Store the best result for each test
            results_df = results_df.append(best_test, ignore_index = True)

            ae_dims_list.pop()

    return results_df


def test_optimised_model(optimised_model, n_tests, classifier):

    # Create an empty series to store the best result
    best_test = pd.Series()

    for i in range(n_tests):

        if model_name not in DL_models_list:
            # Getting the n_components from the optimised models' parameters
            with open(f'{params_path}/{model_name}.json', 'rb') as file:
                params = json.load(file)

            if model_name != 'RFE':
                n_components = int(params['n_components'])
            else:
                n_components = int(params['min_features_to_select'])
        
            # The models are already fit, so we can skip to the reduced dataset generation
            train_X_reduced, test_X_reduced, train_y, test_y = tester.generate_reduced_datasets_nonDL(optimised_model, n_components)

        else:
            train_X_reduced, test_X_reduced, train_y, test_y = tester.generate_reduced_datasets_DL(optimised_model)

        # Fit the classifier
        classifier.fit(train_X_reduced, train_y)

        # Make the predictions
        y_pred = classifier.predict(test_X_reduced)

        # Evaluate the model
        metrics = tester.get_metrics(test_y, y_pred, n_components)

        if best_test.empty:
            best_test = metrics
        if best_test['F1'] < metrics['F1']:
            best_test = metrics
        else:
            pass

    # Replace the model_name by "Opt_{model_name}_model"
    best_test['Model'] = f'Opt_{model_name}_model'

    return best_test


def test_all_variables(classifier):

    # getting the training and testing splits
    train_X, test_X, train_y, test_y = tester.pre_processing()

    # Conduct the testing with the classifier
    y_pred = tester.default_test_preds(classifier)

    # Get the metrics
    metrics = tester.get_metrics(test_y, y_pred, n_components = train_X.shape[1])

    # Replace the model_name by "All_variables"
    metrics['Model'] = 'All_variables'

    return metrics

# --------------------------------------------------------------------------------
# -------------------- ANALYSING THE MODEL(S) TO EVALUATE ------------------------
# --------------------------------------------------------------------------------


# CASE 1 - WE WANT TO CONDUCT TESTS FOR ALL VARIABLES
if len(args) == 1:

    results_df = pd.DataFrame()

    for item in models_list:

        # --------------------------------------------------------------------------------
        # ----------------- CONDUCTING THE TESTING PROCESS FOR THE MODELS ----------------
        # --------------------------------------------------------------------------------
    
        model_name = item
        # instantiating the testing class
        tester = Testing.Testing(model_name)

        # Loading the optimised and default models
        default_model = tester.load_default_model()
        best_model = tester.load_best_model()

        # getting the results of the tests for the default, optimised and all variables models
        default_results = test_default_models(default_model, n_components_list, n_tests, classifier)
        optimised_results = test_optimised_model(best_model, n_tests, classifier)
        all_variables_results = test_all_variables(classifier)

        # Joining all of them into a single dataframe
        if results_df.empty:
            results_df = default_results.append(optimised_results, ignore_index = True)
            results_df = results_df.append(all_variables_results, ignore_index = True)
        
        else:
            results_df = results_df.append(default_results, ignore_index = True)
            results_df = results_df.append(optimised_results, ignore_index = True)
            results_df = results_df.append(all_variables_results, ignore_index = True)

        print(results_df)

        # Saving the results
        results_df.to_csv(f'{Configs["directories"]["results"]}/results_all.csv', index = False)

# CASE 2 - WE WANT TO CONDUCT TESTS FOR A SPECIFIC VARIABLE
else:

    # Analysing the model that was given:
    if len(args) == 2 and args[1].upper() not in models_list:
        raise SyntaxError(f"The model given is not available... Please provide a model to be trained from the following list: {Configs['models']['available_models']}")
    if len(args) > 2:
        raise SyntaxError("Too many arguments were given... Please provide only one model to be trained, or don't provide any if you want to train all the models")
    else: 
        model_name = args[1].upper()


    # --------------------------------------------------------------------------------
    # ----------------- CONDUCTING THE TESTING PROCESS FOR THE MODELS ----------------
    # --------------------------------------------------------------------------------

    # instantiating the testing class
    tester = Testing.Testing(model_name)

    # Loading the optimised and default models
    default_model = tester.load_default_model()
    best_model = tester.load_best_model()

    # getting the results of the tests for the default, optimised and all variables models
    default_results = test_default_models(default_model, n_components_list, n_tests, classifier)
    optimised_results = test_optimised_model(best_model, n_tests, classifier)
    all_variables_results = test_all_variables(classifier)

    # Joining all of them into a single dataframe
    results_df = default_results.append(optimised_results, ignore_index = True)
    results_df = results_df.append(all_variables_results, ignore_index = True)

    # Saving the results
    results_df.to_csv(f'{Configs["directories"]["results"]}/results_{model_name}.csv', index = False)


