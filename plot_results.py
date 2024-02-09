# %%
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  
import json, time, os
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from sklearn.model_selection import cross_val_score
from time import process_time
import exectimeit.timeit
from functools import partial

# Classifiers
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.ensemble import RandomForestClassifier as RF

# Classes
from Classes import Testing

# Conduct tests for the default models, with all the possible number of components
def default_component_tests(classifiers_list):

    # Getting the training and testing datasets
    train_X, test_X, train_y, test_y = tester.pre_processing()

    # Load the default model
    model = tester.load_default_model()
    # Define an array with the possible numer of components
    n_components = np.arange(2, train_X.shape[1], 1)

    ## PLOTING THE RESULTS FROM THE DEFAULT MODELS ##

    # auxiliary df
    aux_df = pd.DataFrame()
    # Iterate over all components for all classifiers
    # for i in n_components:
    for i in n_components:

        # Fitting the DR technique
        tester.fit_DR_technique(model, i)

        # Generate the reduced datasets
        if model_name == 'AE':
            train_X_reduced, test_X_reduced, train_y, test_y = tester.generate_reduced_datasets_DL(model)
        else:
            train_X_reduced, test_X_reduced, train_y, test_y = tester.generate_reduced_datasets_nonDL(model, i)
        
        for classifier in classifiers_list:
            
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

            aux_df = pd.concat([aux_df, metrics], ignore_index = True, axis = 0)

        print("Tested model", model_name, "for", i, "components")

    return aux_df

# Plot a graph with the results from the default model
def plot_default_component_tests(df):

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
        # plt.plot(grouped.get_group(classifiers[1])['N_components'], grouped.get_group(classifiers[1])['MCC'], label = classifiers[1], marker = '^')
        # plt.plot(grouped.get_group(classifiers[2])['N_components'], grouped.get_group(classifiers[2])['MCC'], label = classifiers[2], marker = 's')
        # plt.plot(grouped.get_group(classifiers[3])['N_components'], grouped.get_group(classifiers[3])['MCC'], label = classifiers[3], marker = 'D')

    plt.xlabel("Number of components")
    plt.ylabel("Matthews Correlation Coefficient (MCC)")

    plt.legend()
    
    if not os.path.exists('./Results/Plots/Default_Tests'):
        os.makedirs('./Results/Plots/Default_Tests')
    plt.savefig(f'./Results/Plots/Default_Tests/{model_name}_default_tests.svg', dpi = 300)

# Save the default models' results to a csv file
def save_default_component_tests(df, classifiers_names):
    
    # Grouping the dataframes per classifier
    grouped = df.groupby('Classifier')

    # Creating a directory, if it doesn't exist, for this results
    if not os.path.exists('./Results/Default_Tests'):
        os.makedirs('./Results/Default_Tests') 

    for classifier in classifiers_names:
        # Get the group intended
        group = grouped.get_group(classifier)
        # Store the dataframe in the specific path
        group.to_csv(f'./Results/Default_Tests/{model_name}_{classifier}.csv')

# Testing the optimised model
def test_optimised_model(classifiers_list):

    aux_df = pd.DataFrame()

    # Load the optimised model
    optimised_model = tester.load_best_model()


    params_path = './Models/Parameters'
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
        
        aux_df = pd.concat([aux_df, metrics], ignore_index = True, axis = 0)

    return aux_df

# Bar plot for the 4 classifiers - optimised model results
def plot_test_optimised_model(df):

    custom_palette = sns.color_palette("Set1")  # You can choose from various predefined palettes
    # Set the style and palette for the entire plot
    sns.set(style = 'ticks' ,palette = custom_palette)

    plt.figure(figsize=(8,6))

    plt.bar(df['Classifier'], df['F1'], width=0.5)
    plt.legend()
    plt.ylabel('Matthews Correlation Coefficient (MCC)')

    for i, value in enumerate(df['F1']):
        plt.text(i, value + 0.01, str(round(value,3)), ha='center', va='bottom', fontsize=8)
        
    plt.show()

# Save the optimised models' results to a csv file
def save_test_optimised_model(df, classifiers_list):
    
    # Grouping the dataframes per classifier
    grouped = df.groupby('Classifier')

    # Creating a directory, if it doesn't exist, for this results
    if not os.path.exists('./Results/Optimised_Tests'):
        os.makedirs('./Results/Optimised_Tests') 

    for classifier in classifiers_list:

        # Get the group intended
        group = grouped.get_group(classifier)
        # Store the dataframe in the specific path
        group.to_csv(f'./Results/Optimised_Tests/{classifier}.csv')
  
# Radar chart "class"
def radar_chart(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    
    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_variables(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                    radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def draw(self, renderer):
            """ Draw. If frame is polygon, make gridlines polygon-shaped """
            if frame == 'polygon':
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)


        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                            spine_type='circle',
                            path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)


                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

# Plot the radar chart for 1 particular test
def plot_radar_chart(df):

    # NOTE: Code from https://stackoverflow.com/questions/52910187/how-to-make-a-polygon-radar-spider-chart-in-python
    # auxliary variable to plot the graph
    n_sides = 5 # F1, Recall, Accuracy, Precision, Specificity
    
    # Start the graph
    theta = radar_chart(n_sides, frame='polygon')

    graph_labels = ['F1', 'Recall', 'Accuracy', 'Precision', 'Specificity']
    title = f'Model {model_name}'
    data = df[['F1', 'Recall', 'Accuracy', 'Precision', 'Specificity', 'Classifier']]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(top=0.85, bottom=0.05)

    for row in data.index: 
        values = data.iloc[row, :-1].values
        classifier_name = data.iloc[row, -1]
        line = ax.plot(theta, values, label=classifier_name, linewidth=2)
        # ax.fill(theta, values, alpha=0.25, label=classifier_name)
    
    ax.set_variables(graph_labels)
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8], angle = 0)

    plt.legend()
    plt.show()

# Conduct multiple tests for a certain model, with a certain number of components
def multiple_tests(model_type: str, classifiers_list, n_components=None, n_tests = 5):

    aux_df = pd.DataFrame()

    # Getting the training and testing datasets
    train_X, test_X, train_y, test_y = tester.pre_processing()

    if model_type == 'default':
        # Load the default model
        model = tester.load_default_model()
    else:
        # Load the optimised model
        model = tester.load_best_model()
    
    # Fitting the DR technique
    tester.fit_DR_technique(model, n_components)
    
    # Generate the reduced datasets
    if model_name == 'AE':
        train_X_reduced, test_X_reduced, train_y, test_y = tester.generate_reduced_datasets_DL(model)
    else:
        train_X_reduced, test_X_reduced, train_y, test_y = tester.generate_reduced_datasets_nonDL(model, n_components)

    for classifier in classifiers_list:

        for i in range(n_tests):

            # Fit the classifier
            classifier.fit(train_X_reduced, train_y)
            # Make the predictions
            y_pred = classifier.predict(test_X_reduced)
            # Evaluate the model
            metrics = tester.get_metrics(test_y, y_pred, n_components)
            metrics.drop('Model', inplace = True)
            metrics['Classifier'] = classifier.__class__.__name__
            aux_df = pd.concat([aux_df, metrics], ignore_index = True, axis = 0)

            print("Test", i, "for", classifier.__class__.__name__, "with", n_components, "components concluded")

        print("Tested model", model_name, "for", n_components, "components")

    return aux_df

# Conduct multiple tests to assess the training and testing times for each classifier - depending on the number of components
def test_times(classifiers_list, n_tests = 5):

    # Getting the training and testing datasets
    train_X, test_X, train_y, test_y = tester.pre_processing()
    # Defining a set of number of components
    n_components = np.arange(2, train_X.shape[1] + 1, 1)
    # auxiliar dataframe
    results_df = pd.DataFrame()
    # Load the default model
    dr_model = tester.load_default_model()

    for classifier in classifiers_list:

        for components in n_components:

            # Fitting the DR technique
            tester.fit_DR_technique(dr_model, components)

            # Generate the reduced datasets
            if model_name == 'AE':
                train_X_reduced, test_X_reduced, train_y, test_y = tester.generate_reduced_datasets_DL(dr_model)
            else:
                train_X_reduced, test_X_reduced, train_y, test_y = tester.generate_reduced_datasets_nonDL(dr_model, components)

            classifier_name = classifier.__class__.__name__

            # Defining a function that fits the model
            def model_fit(X, y, classifier):
                classifier.fit(X, y)
                # return classifier

            # Defining a function that predicts the model
            def model_pred(X, classifier):
                preds = classifier.predict(X)
                return preds
            
            # Creating a function with fixed arguments
            model_fit = partial(model_fit, train_X_reduced, train_y, classifier)
            # Measuring the time to fit the model
            mean_train_time, variation, classifier = exectimeit.timeit.timeit(n_tests, model_fit)
            print(f"Average time to fit the classifier {classifier.__class__.__name__} with {components} variables using {model_name} technique: {mean_train_time*1000} ms")
            
            # Creating a function with fixed arguments
            model_pred = partial(model_pred, test_X_reduced, classifier)
            # Measuring the time to make the predictions on test data
            mean_pred_time, variation, classifier = exectimeit.timeit.timeit(n_tests, model_pred)
            print(f"Average time to make the predictions for the classifier {classifier.__class__.__name__} with {components} variables using {model_name} technique: {mean_pred_time*1000} ms")

            # Storing the times within a dataframe
            times = pd.Series([classifier_name, components, mean_train_time*1000, mean_pred_time], index = ['Classifier', 'N_components', 'Training time', 'Prediction time'])
            results_df= pd.concat([results_df, times.to_frame().T], ignore_index = False, axis = 0)

                # time.sleep(1)
    # # Storing this dataframe
    # if not os.path.exists('./Results/Time_Tests'):
    #     os.makedirs('./Results/Time_Tests')
    # results_df.to_csv(f'./Results/Time_Tests/{model_name}_Times.csv', index = False)

    # print(results_df)

    return results_df

# Function that plots the results from the time tests
def plot_times(results_df, classifiers_names):

    # Storing the time tests on a specific folder
    if not os.path.exists('./Results/Times'):
        os.makedirs('./Results/Times')
    if not os.path.exists('./Results/Plots/Times'):
        os.makedirs('./Results/Plots/Times')

    # Removing first column that is not needed
    # results_df.drop('Attempt no.', axis = 1, inplace = True)
    # Storing also this dataframe with all info
    results_df.to_csv(f'./Results/Times/{model_name}_all_times.csv')

    # grouping the dataframes
    grouped = results_df.groupby(['Classifier', 'N_components'])
    # Getting their average metrics values
    avg_grouped = grouped.mean()  # Mean + ms conversion

    plot_markers = ['o', '^', 's', 'D']
    # Plotting the results    
    # plot a graph with the training times
    for count, name in enumerate(classifiers_names):
        plt.subplot(len(classifiers_names), 1, count + 1)
        plt.plot(avg_grouped.loc[name].index.values, avg_grouped.loc[name].loc[:, 'Training time'].values, marker = plot_markers[count], label = name)
        plt.legend()
        plt.xlabel('Number of components')
        plt.ylabel('Training time (ms)')
    plt.savefig(f'./Results/Plots/Times/{model_name}_train_times.svg', dpi = 300)  
    
    # plot a graph with the prediction times
    for count, name in enumerate(classifiers_names):
        plt.subplot(len(classifiers_names), 1, count + 1)
        plt.plot(avg_grouped.loc[name].index.values, avg_grouped.loc[name].loc[:, 'Prediction time'].values, marker = plot_markers[count], label = name)
        plt.legend()
        plt.xlabel('Number of components')
        plt.ylabel('Prediction time (ms)')
    plt.savefig(f'./Results/Plots/Times/{model_name}_pred_times.svg', dpi = 300)   

    for classifier in classifiers_names:
        # to_save_predictions = avg_grouped.loc[classifier].loc[:, 'Prediction time']
        to_save_training = avg_grouped.loc[classifier].loc[:, 'Training time']
        # Saving both dataframes
        # to_save_predictions.to_csv(f'./Results/Times/{model_name}_{classifier}_pred_times.csv')
        to_save_training.to_csv(f'./Results/Times/{model_name}_{classifier}_train_times.csv')
# %%


if __name__ == '__main__':
   
    # Instantiate all classifiers
    mlp = MLP()
    logreg = LogReg()
    knn = kNN()
    rf = RF()
    
    # generate a list with the classifiers we wish to test
    classifiers_list = [knn, logreg, rf, mlp]
    classifiers_names = [x.__class__.__name__ for x in classifiers_list]

    # Choosing the DR technique
    dr_techniques = ['PCA', 'ICA', 'SVD', 'RF', 'NMF', 'RFE', 'LDA']
    # model_name = 'RF'

    # for model_name in dr_techniques:

    #     # instantiating the class
    #     tester = Testing.Testing(model_name)

    #     # Conducting the default model tests
    #     default_df = default_component_tests(classifiers_list)
    #     plot_default_component_tests(default_df)
    #     save_default_component_tests(default_df, classifiers_names)

    #     # Conducting the optimised model tests
    #     optimised_df = test_optimised_model(classifiers_list)
    #     save_test_optimised_model(optimised_df, classifiers_names)

    # #     print(f'Tests for DR technique {model_name} concluded... Moving to the next one... \n\n')
    
    # Getting the training and testing time for 1 DR technique - PCA
    model_name = 'PCA'    
    tester = Testing.Testing(model_name)
    # Conducting the time tests
    times_df = test_times(classifiers_list, n_tests = 10)
    plot_times(times_df, classifiers_names)

   
