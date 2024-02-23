# %%
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  
import json, os
import tqdm, yaml
import exectimeit
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Classifiers
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.ensemble import RandomForestClassifier as RF

# Classes
from Classes import Testing

# Importing some initial settings
with open('settings.yaml') as file:
    Configs = yaml.full_load(file)
validation_split = Configs['models']['validation_split']
epochs = Configs['models']['num_final_epochs']
# Some important paths
data_path = Configs['directories']['datasets']
default_tests_path = Configs['directories']['default_tests']
time_tests_path = Configs['directories']['time_tests']
dr_time_tests_path = Configs['directories']['dr_time_tests']
params_path = Configs['directories']['parameters']


# Conduct tests for the default models, with all the possible number of components
def default_component_tests(classifiers_list):

    """ We take as input a list of a series of classifiers, the list being comprised
    of the models themselves, and then iterate over all the possible number of components
    
    For the Autoencoder, a diferent process is conducted, as we need to build the Autoencoder
    to test according to a set of dimensions we wish to use
    
    The function returns a DataFrame with the results for each test"""


    # Getting the training and testing datasets
    train_X, test_X, train_y, test_y = tester.pre_processing()
    # Load the default model
    model = tester.load_default_model()
    # Define an array with the possible numer of components
    n_components = np.arange(2, train_X.shape[1], 1)

    # auxiliary df
    aux_df = pd.DataFrame()

    # Process if we have an autoencoder
    if model_name == 'AE':
        
        dims = [[train_X.shape[1], 32, 16, 8, 4, 2],
                [train_X.shape[1], 32, 16, 8, 4],
                [train_X.shape[1], 32, 16, 8],
                [train_X.shape[1], 32, 16],
                [train_X.shape[1], 32]]
        
        for dim in tqdm.tqdm(dims):
            
            # Build the autoencoder
            ae = tester.build_ae(dim)

            # Fitting the autoencoder
            ae = tester.fit_DR_technique(ae, dim)
            encoder = tf.keras.Model(ae.input, ae.layers[-2].output)
            print(encoder.summary())

            # Generate the reduced datasets
            train_X_reduced, test_X_reduced, train_y, test_y = tester.generate_reduced_datasets_DL(encoder)

            for classifier in tqdm.tqdm(classifiers_list, leave = False):
                
                # Get the classifier's name
                classifier_name = classifier.__class__.__name__
                # Fitting the classifier
                classifier.fit(train_X_reduced, train_y)
                # Make the predictions
                y_pred = classifier.predict(test_X_reduced)
                # Evaluate the model - get a dataframe with the metrics for each class of data (we have 4 classes, so each dataframe as 4 rows, 1 relative to each class)
                metrics = tester.get_metrics(test_y, y_pred, dim[-1])
                # metrics.drop('Model', inplace = True)
                metrics['Classifier'] = classifier_name

                aux_df = pd.concat([aux_df, metrics.to_frame().T], ignore_index = True, axis = 0)
   
    # Process if we have any other model
    else:
        # Iterate over all components for all classifiers
        for i in tqdm.tqdm(n_components):

            # Fitting the DR technique
            tester.fit_DR_technique(model, i)

            train_X_reduced, test_X_reduced, train_y, test_y = tester.generate_reduced_datasets_nonDL(model, i)
            
            for classifier in tqdm.tqdm(classifiers_list):
                
                # Get the classifier's name
                classifier_name = classifier.__class__.__name__
                # Fitting the classifier
                classifier.fit(train_X_reduced, train_y)
                # Make the predictions
                y_pred = classifier.predict(test_X_reduced)
                # Evaluate the model - get a dataframe with the metrics for each class of data (we have 4 classes, so each dataframe as 4 rows, 1 relative to each class)
                metrics = tester.get_metrics(test_y, y_pred, i)
                # metrics.drop('Model', inplace = True)
                metrics['Classifier'] = classifier_name

                aux_df = pd.concat([aux_df, metrics.to_frame().T], ignore_index = True, axis = 0)


    # Testing with all the features - Default test basically
    for classifier in tqdm.tqdm(classifiers_list):
        # Get the classifier's name
        classifier_name = classifier.__class__.__name__
        # Fitting the classifier
        classifier.fit(train_X, train_y)
        # Make the predictions
        y_pred = classifier.predict(test_X)
        # Evaluate the model - get a dataframe with the metrics for each class of data (we have 4 classes, so each dataframe as 4 rows, 1 relative to each class)
        metrics = tester.get_metrics(test_y, y_pred, train_X.shape[1])
        # metrics.drop('Model', inplace = True)
        metrics['Classifier'] = classifier_name

        aux_df = pd.concat([aux_df, metrics.to_frame().T], ignore_index = True, axis = 0)

    return aux_df

# Plot a graph with the results from the default model
def plot_default_component_tests(df):

    """ Function that just plots a graph of the results from the default tests and
    stores ir under a specific path"""

    # Group the dataframe by the classifier
    grouped = df.groupby('Classifier')
    # different classifiers
    classifiers = df['Classifier'].unique().tolist()

    custom_palette = sns.color_palette("Set1")  # You can choose from various predefined palettes

    # Set the style and palette for the entire plot
    sns.set(style = 'ticks' ,palette = custom_palette[0:4])

    plt.figure(figsize=(12,8))
    plot_markers = ['o', '^', 's', 'D']
    # Plotting the results
    for value, classifier in enumerate(classifiers):
        plt.plot(grouped.get_group(classifier)['N_components'], grouped.get_group(classifier)['MCC'], label = classifier, marker = plot_markers[value])

    plt.xlabel("Number of components")
    plt.ylabel("Matthews Correlation Coefficient (MCC)")

    plt.legend()
    
    if not os.path.exists('./Results/Plots/Default_Tests'):
        os.makedirs('./Results/Plots/Default_Tests')
    plt.savefig(f'./Results/Plots/Default_Tests/{model_name}_default_tests.svg', dpi = 300)

# Save the default models' results to a csv file
def save_default_component_tests(df, classifiers_names):
    
    """ Function to simply store, in an organised fashion, the results from the 
    default tests for each classifier """

    # Grouping the dataframes per classifier
    grouped = df.groupby('Classifier')

    # Creating a directory, if it doesn't exist, for this results
    if not os.path.exists(default_tests_path):
        os.makedirs(default_tests_path) 

    for classifier in classifiers_names:
        # Get the group intended
        group = grouped.get_group(classifier)
        # Store the dataframe in the specific path
        group.to_csv(f'{default_tests_path}/{model_name}_{classifier}.csv')

# Testing the optimised model
def test_optimised_model(classifiers_list):

    """ Function that takes as input a list of classifiers, and then tests
    the models that had their hyperparameters optimised. The funtion returns a 
    dataframe with the results for each test"""

    aux_df = pd.DataFrame()
    # Load the optimised model
    optimised_model = tester.load_best_model()

    if model_name != 'AE':
        # Getting the n_components from the optimised models' parameters
        with open(f'{params_path}/{model_name}.json', 'rb') as file:
            params = json.load(file)

        if model_name != 'RFE':
            n_components = int(params['n_components'])
        else:
            n_components = int(params['min_features_to_select'])
        # Getting the training and testing datasets
        train_X_reduced, test_X_reduced, train_y, test_y = tester.generate_reduced_datasets_nonDL(optimised_model, n_components)

    else:
        n_components = optimised_model.layers[-1].output_shape[1] 
        train_X_reduced, test_X_reduced, train_y, test_y = tester.generate_reduced_datasets_DL(optimised_model)
        

    for classifier in classifiers_list:
        
        # Get the classifier's name
        classifier_name = classifier.__class__.__name__
        # Fitting the classifier
        classifier.fit(train_X_reduced, train_y)
        # Make the predictions
        y_pred = classifier.predict(test_X_reduced)
        # Evaluate the model
        metrics = tester.get_metrics(test_y, y_pred, n_components)
        # metrics.drop('Model', inplace = True)
        metrics['Classifier'] = classifier_name
        
        aux_df = pd.concat([aux_df, metrics.to_frame().T], ignore_index = True, axis = 0)

    return aux_df

# Save the optimised models' results to a csv file
def save_test_optimised_model(df, classifiers_list):
    
    """ Function to simply store, in an organised fashion, the results from the 
    optimised tests for each classifier """
        
    # Grouping the dataframes per classifier
    grouped = df.groupby('Classifier')

    # Creating a directory, if it doesn't exist, for this results
    if not os.path.exists('./Results/Optimised_Tests'):
        os.makedirs('./Results/Optimised_Tests') 

    for classifier in classifiers_list:

        # Get the group intended
        group = grouped.get_group(classifier)
        # Store the dataframe in the specific path
        group.to_csv(f'./Results/Optimised_Tests/{model_name}_{classifier}.csv')
  
# Testing the classifier times
def test_classifier_times(classifiers_list, n_tests = 5):

    """ Function used to test the time it takes for a classifier to fit to the train
    data and to predict the test data, using the reduced datasets
    
    It has 2 inputs:
        - classifiers_list: a list with the classifiers we wish to test
        - n_tests: the number of tests we wish to conduct (default = 5) that is input to the
        exectimit library - each value returned by the exectimeit.timeit funtion
        is the average for those 5 tests
        
    It returns a dataframe with the average results for the 'n_tests' conducted, 
    alongisde their std values """

    # Getting the training and testing datasets
    train_X, test_X, train_y, test_y = tester.pre_processing()
          
    # Defining a set of number of components
    # n_components = np.arange(2, train_X.shape[1], 1)
    n_components = [2, 5, 10, 20, 30, 40]

    results_df = pd.DataFrame()

    # Load the default model
    dr_model = tester.load_default_model()

    for classifier in tqdm.tqdm(classifiers_list):

        classifier_name = classifier.__class__.__name__

        for components in tqdm.tqdm(n_components, leave = False):

            # Fitting the DR technique
            tester.fit_DR_technique(dr_model, components)

            # Generate the reduced datasets
            if model_name == 'AE' or model_name == 'NMF':
                train_X_reduced, test_X_reduced, train_y, test_y = tester.generate_reduced_datasets_DL(dr_model)
            else:
                train_X_reduced, test_X_reduced, train_y, test_y = tester.generate_reduced_datasets_nonDL(dr_model, components)

            # Average time for the fitting
            avg_train_time, avg_train_std, _ = exectimeit.timeit.timeit(n_tests, classifier.fit, train_X_reduced, train_y)
            # Average time for the predictions
            avg_pred_time, avg_pred_std, _ = exectimeit.timeit.timeit(n_tests, classifier.predict, test_X_reduced)

            # Storing the times within a dataframe
            times = pd.Series([classifier_name, components, avg_train_time*1000, avg_train_std*1000, avg_pred_time*1000, avg_pred_std*1000], index = ['Classifier', 'N_components', 'Training time', 'Training std', 'Testing time', 'Testing std'])
            results_df= pd.concat([results_df, times.to_frame().T], ignore_index = False, axis = 0)

    # Storing this dataframe
    if not os.path.exists(time_tests_path):
        os.makedirs(time_tests_path)
    results_df.to_csv(f'{time_tests_path}/{model_name}_Times.csv', index = False)

    return results_df

def default_classifier_times(classifiers_list, n_tests = 5):

    """ Function used to test the time it takes for a classifier to fit to the train
    data and to predict the test data, with the entire original dataset
    
    It has 2 inputs:
        - classifiers_list: a list with the classifiers we wish to test
        - n_tests: the number of tests we wish to conduct (default = 5) that is input to the
        exectimit library - each value returned by the exectimeit.timeit funtion
        is the average for those 5 tests    
        
    It returns a dataframe with the average results for the 'n_tests' conducted, 
    alongisde their std values """

    # Loading the dataset
    X = pd.read_csv(f'{data_path}/X_StandardScaler.csv')
    y = pd.read_csv(f'{data_path}/y_LabelEncoder.csv')

    # Generating the training and testing datasets
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 42)

    results_df = pd.DataFrame()

    for classifier in tqdm.tqdm(classifiers_list):

        classifier_name = classifier.__class__.__name__

        # Average time for the fitting
        avg_train_time, avg_train_std, _ = exectimeit.timeit.timeit(n_tests, classifier.fit, train_X, train_y)
        classifier.fit(train_X, train_y)
        # Average time for the predictions
        avg_pred_time, avg_pred_std, _ = exectimeit.timeit.timeit(n_tests, classifier.predict, test_X)

        # Storing the times within a dataframe
        times = pd.Series([classifier_name, avg_train_time*1000, avg_train_std*1000, avg_pred_time*1000, avg_pred_std*1000], index = ['Classifier', 'Training time', 'Training std', 'Testing time', 'Testing std'])
        results_df= pd.concat([results_df, times.to_frame().T], ignore_index = False, axis = 0)

    # Further processing the DataFrame
    final_df = pd.DataFrame()
    for row in results_df.iterrows():

        row = row[1]
        classifier = row['Classifier']
        a = round(float(row['Training time']), 3)
        b = round(float(row['Training std']), 3)

        c = round(float(row['Testing time']), 3)
        d = round(float(row['Testing std']), 3)

        aux_series = pd.Series([classifier, str(a) + '+' + str(b), str(c) + '+' + str(d)], index = ['Classifier', 'Training time', 'Testing time'])
        final_df = pd.concat([final_df, aux_series.to_frame().T], axis = 0)
    
    # Storing this dataframe
    if not os.path.exists(time_tests_path):
        os.makedirs(time_tests_path)
    final_df.to_csv(f'{time_tests_path}/Default_Classifier_Times.csv', index = False)

    return final_df

# # Function that plots the results from the time tests
# def plot_classifier_times(results_df, classifiers_names):

#     # Storing the time tests on a specific folder
#     if not os.path.exists('./Results/Times'):
#         os.makedirs('./Results/Times')
#     if not os.path.exists('./Results/Plots/Times'):
#         os.makedirs('./Results/Plots/Times')

#     # grouping the dataframes
#     grouped = results_df.groupby(['Classifier'])

#     plot_markers = ['o', '^', 's', 'D']
#     # Plotting the results    
#     # plot a graph with the training times
#     for count, name in enumerate(classifiers_names):
#         plt.subplot(len(classifiers_names), 1, count + 1)
#         plt.plot(grouped.get_group(name).loc[:, 'N_components'], grouped.get_group(name).loc[:, 'Training time'], marker = plot_markers[count], label = name)
#         plt.legend()
#         plt.xlabel('Number of components')
#         plt.ylabel('Training time (ms)')
#     plt.savefig(f'./Results/Plots/Times/{model_name}_train_times.svg', dpi = 300)  
#     plt.clf()

#     # plot a graph with the prediction times
#     for count, name in enumerate(classifiers_names):
#         plt.subplot(len(classifiers_names), 1, count + 1)
#         plt.plot(grouped.get_group(name).loc[:, 'N_components'], grouped.get_group(name).loc[:, 'Testing time'].values, marker = plot_markers[count], label = name)
#         plt.legend()
#         plt.xlabel('Number of components')
#         plt.ylabel('Prediction time (ms)')
#     plt.savefig(f'./Results/Plots/Times/{model_name}_pred_times.svg', dpi = 300)   

#     for classifier in classifiers_names:
#         # Get the group intended
#         group = grouped.get_group(classifier)
#         # Store the dataframe in the specific path
#         group.to_csv(f'./Results/Default_Tests/{model_name}_{classifier}_Times.csv')

# Function that assesses the time for a certain DR technique to fit and predict the data
def test_DR_technique_times(n_tests = 5):

    """ Function used to test the time it takes for a DR technique to first fit
    to the training data, and then to reduce the input dataset
    
    It only needs as input the number of tests we wish to conduct for each DR technique
    with a certain classifier
        
    It returns a dataframe with the average results for the 'n_tests' conducted, 
    alongisde their std values """

    # Getting the training and testing datasets
    train_X, test_X, train_y, test_y = tester.pre_processing()
    # Auxiliary dataframe
    results_df = pd.DataFrame()

    if model_name == 'AE':
        # Defining a set of number of components
        dims = [[train_X.shape[1], 32, 16, 8, 4, 2],
                        [train_X.shape[1], 32, 16, 8, 4],
                        [train_X.shape[1], 32, 16, 8],
                        [train_X.shape[1], 32, 16],
                        [train_X.shape[1], 32]]
        
        for dim in tqdm.tqdm(dims):

            # Build the autoencoder
            ae = tester.build_ae(dim)
            # Assessing the fitting time for the autoencoder
            fit_time, fit_std, _ = exectimeit.timeit.timeit(n_tests, ae.fit, train_X, train_X, epochs = epochs, validation_split = validation_split, shuffle = True)
            # Getting the encoder
            encoder = tf.keras.Model(ae.input, ae.layers[-2].output)
            print(encoder.summary())

            # Generate the reduced datasets
            predict_train_time, predict_train_std, _ = exectimeit.timeit.timeit(n_tests, ae.predict, train_X)
            predict_test_time, predict_test_std, _ = exectimeit.timeit.timeit(n_tests, ae.predict, test_X)

            # Storing the times within a dataframe
            # NOTE: Should I store the fitting time per epoch?
            times = pd.Series([model_name, component, fit_time*1000, fit_std*1000, predict_train_time*1000, predict_train_std*1000, predict_test_time*1000, predict_test_std*1000], index = ['DR Model', 'N_components', 'Fit time', 'Fit std', 'Pred train time', 'Pred train std std', 'Pred test time', 'Pred test std'])
            results_df = pd.concat([results_df, times.to_frame().T], ignore_index = False, axis = 0)

    
    # Tests only conducted for the feature extraction models
    elif model_name not in Configs['models']['FS_models']:

        # Load the default model
        dr_model = tester.load_default_model()
        # Defining a set of number of components (all but the max possible)
        n_components = np.arange(2, train_X.shape[1], 1)
   
        for component in tqdm.tqdm(n_components):
            # Setting the number of components for the model, if possible
            try:
                dr_model.set_params(n_components = component)
            except:
                pass

            # Fitting times for feature extraction techniques
            fit_time, fit_std, _ = exectimeit.timeit.timeit(n_tests, dr_model.fit, train_X)
            # Prediction times for feature extraction techniques
            predict_train_time, predict_train_std, _ = exectimeit.timeit.timeit(n_tests, dr_model.transform, train_X)
            predict_test_time, predict_test_std, _ = exectimeit.timeit.timeit(n_tests, dr_model.transform, test_X)

            # Storing the times within a dataframe
            times = pd.Series([model_name, component, fit_time*1000, fit_std*1000, predict_train_time*1000, predict_train_std*1000, predict_test_time*1000, predict_test_std*1000], index = ['DR_Model', 'N_components', 'Fit time', 'Fit std', 'Pred train time', 'Pred train std', 'Pred test time', 'Pred test std'])
            results_df= pd.concat([results_df, times.to_frame().T], ignore_index = False, axis = 0)
    else:
        pass

    # Storing this dataframe
    if not os.path.exists(dr_time_tests_path):
        os.makedirs(dr_time_tests_path)
    results_df.to_csv(f'{dr_time_tests_path}/{model_name}_DR_Times.csv', index = False)

    return results_df


# # Function that plots the results from the fitting and prediction times of the DR techniques
# def plot_DR_technique_times(results_df):

#     # Storing the time tests on a specific folder
#     if not os.path.exists('./Results/Times'):
#         os.makedirs('./Results/Times')
#     if not os.path.exists('./Results/Plots/Times_DR'):
#         os.makedirs('./Results/Plots/Times_DR')

#     plt.subplot(2, 1, 1)
#     plt.plot(results_df['N_components'], results_df['Fit time'], marker = 'o', color = 'b')
#     # plt.errorbar(results_df['N_components'], results_df['Pred train time'], yerr = results_df['Pred train std'], fmt = 'o', color = 'b')
#     plt.xlabel('Number of components')
#     plt.ylabel('DR Training time (ms)') 
    
#     plt.subplot(2, 1, 2)
#     plt.plot(results_df['N_components'], results_df['Pred test time'], marker = 'o', color = 'b', label = 'Test')
#     # plt.errorbar(results_df['N_components'], results_df['Pred test time'], yerr = results_df['Pred test std'], fmt = 'o', color = 'b')
#     plt.xlabel('Number of components')
#     plt.ylabel('DR test time (ms)')

#     plt.savefig(f'./Results/Plots/Times_DR/{model_name}_times.svg', dpi = 300)  

# %%


if __name__ == '__main__':
   
    # Instantiate all classifiers
    mlp = MLP(max_iter = 5000)
    logreg = LogReg(max_iter = 5000)
    knn = kNN()
    rf = RF()
    
    # generate a list with the classifiers we wish to test
    classifiers_list = [logreg, knn, rf, mlp]
    classifiers_names = [x.__class__.__name__ for x in classifiers_list]

    # Choosing the DR technique
    dr_techniques = Configs['models']['available_models']
    # model_name = 'RF'

    # Conducting the performance tests
    for dr_technique in dr_techniques:

        # # instantiating the class
        tester = Testing.Testing(dr_technique)

        # # Conducting the default model tests
        default_df = default_component_tests(classifiers_list)
        plot_default_component_tests(default_df)
        save_default_component_tests(default_df, classifiers_names)

        # Conducting the optimised model tests
        optimised_df = test_optimised_model(classifiers_list)
        save_test_optimised_model(optimised_df, classifiers_names)


    # 2. Conducting the classifier time testing for all DR techniques
    for dr_technique in dr_techniques:

        # instantiating the class
        tester = Testing.Testing(dr_technique)

        # Conducting the classifier time tests
        times_df = test_classifier_times(classifiers_list, n_tests = 5)
        print(f'Tests for DR technique {dr_technique} concluded... Moving to the next one... \n\n')


    # 3. Some DR methods' time testing
    time_tests = ['PCA', 'NMF', 'AE']    
    for dr_technique in time_tests:
        
        # instantiating the class
        tester = Testing.Testing(dr_technique)

        # Conducting the DR technique times tests
        dr_times_df = test_DR_technique_times(n_tests = 5)
        print(f'Time tests for DR technique {dr_technique} concluded... Moving to the next one... \n\n')


    # 4. Optimised model testing and saving the results    
    for model_name in dr_techniques:

        # instantiating the class
        tester = Testing.Testing(model_name)

        optimised_df = test_optimised_model(classifiers_list)
        save_test_optimised_model(optimised_df, classifiers_names)
   

# %%
