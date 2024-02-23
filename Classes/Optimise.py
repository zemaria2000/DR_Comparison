""" This is a class that was defined to help the model hyperparameter optimisation process

In the paper of this project, the hyperparameter is not present, as it wasn't fully developed
and the results were not satisfactory. However, the optimisation code is available here"""


import numpy as np 
from sklearn.model_selection import train_test_split
import os, json, yaml, pickle
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Scikit-Optimisation library
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from skopt.utils import use_named_args
from skopt import gp_minimize

# Scikit-Learn DR and classification models
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.decomposition import TruncatedSVD as SVD
from sklearn.feature_selection import RFECV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier as RF

# Tensroflow and keras libraries
import tensorflow as tf 
import keras_tuner as kt
from keras_tuner import HyperParameters as hp
from tensorflow.keras import backend as K


class hyperparameter_optimiser:

    def __init__(self, model_name: str):

        """ The class' constructor method, which will load all the settings and parameters
        and define the variables that will be used throughout the class' methods"""

        # First, load all the settings and parameters
        with open('settings.yaml') as file:
            Configs = yaml.full_load(file)

        # Defining a series of variables according to the settings
        self.models_path, self.parameters_path, self.data_path, self.search_space_path, self.opt_results = Configs['directories']['best_model'], Configs['directories']['parameters'], Configs['directories']['datasets'], Configs['directories']['search_space'], Configs['directories']['opt_results']
        # Variables related to the models' training
        self.trial_epochs, self.num_trials, self.epochs = Configs['models']['num_epochs_trials'], Configs['models']['num_trials'], Configs['models']['num_final_epochs']
        self.train_split, self.validation_split = Configs['models']['train_split'], Configs['models']['validation_split']
        # X and y datasets
        self.X_DL = np.loadtxt(f'{self.data_path}/X_MinMaxScaler.csv', delimiter = ',')
        self.X_NonDL = np.loadtxt(f'{self.data_path}/X_StandardScaler.csv', delimiter = ',')
        self.y = np.loadtxt(f'{self.data_path}/y_LabelEncoder.csv', delimiter = ',')
        # Getting a list of the DL models (and all the models)
        self.DL_models, self.FS_models, self.available_models = Configs['models']['DL_models'], Configs['models']['FS_models'], Configs['models']['available_models']
        # Defining the model's name
        self.model_name = model_name

        # Defining a callback that saves our model - in case of a DL tensorflow built model
        self.cp = tf.keras.callbacks.ModelCheckpoint(filepath = f'{self.models_path}/{self.model_name}.h5',
                                        mode = 'min', monitor = 'val_loss', verbose = 2 , save_best_only = True)
        
        # Defining a callback that stops the search if the results aren't improving - DL tensorflow models
        self.early_stop = tf.keras.callbacks.EarlyStopping(
            monitor = 'val_loss',
            min_delta = 0.0001,
            patience = 20,
            verbose = 1, 
            mode = 'min',
            restore_best_weights = True)


    # Instantiating the model when it is not a DL model
    def define_model(self):

        """ Method that defines the model to be trained, according to the model's name"""

        # Checking the model to be trained
        if self.model_name == 'PCA':
            self.model = PCA()
        elif self.model_name == 'ICA':
            self.model = FastICA()
        elif self.model_name == 'SVD':
            self.model = SVD()
        elif self.model_name == 'LDA':
            self.model = LDA()
        elif self.model_name == 'RF':
            self.model = RF()
        elif self.model_name == 'NMF':
            self.model = NMF()
        elif self.model_name == 'RFE':
            self.model = RFECV(estimator = LogisticRegression())


    # Creating the training and testing splits
    def pre_processing(self):

        """ Method that creates the training and testing splits, according to the model's name"""

        if self.model_name in self.DL_models or self.model_name == 'NMF':
            # Returning the train and test splits
            return train_test_split(self.X_DL, self.y, test_size = 1-self.train_split, random_state = 42, shuffle = True)
        else:
            return train_test_split(self.X_NonDL, self.y, test_size = 1-self.train_split, random_state = 42, shuffle = True)
        

    # Function that optimises the autoencoder's hyperparameters
    def build_optimised_autoencoder(self):

        """ Method that optimises the hyperparameters of an autoencoder, using the keras-tuner library"""

        def tuning_ae(hp):

            ## DEFINING THE HYPERPARAMETERS TO BE TUNED
            # Number of hiddewn layers
            self.n_hidden = hp.Int('Hidden_Layers', min_value = 3, max_value = 7)
            # Drop between each layer, which will define the size of the subsequent layer
            self.layers_drop = []
            for i in range(self.n_hidden):
                self.layers_drop.append(hp.Float(f"drop_{i}-{i+1}", min_value = 1.2, max_value = 1.8))
            # Layer dimensions, which depend on drop between layers
            self.layers_dims = []
            for i in range(self.n_hidden):
                if i == 0:      # first layer
                    self.layers_dims.append(int(self.X_DL.shape[1]/self.layers_drop[i]))
                else:
                    self.layers_dims.append(int(self.layers_dims[i-1]/self.layers_drop[i]))
            # Activation function - https://keras.io/2.15/api/layers/activations/
            self.activation_function = hp.Choice('Activation_Function', values = ['relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh', 'selu', 'elu'])
            # # Optimizer - https://keras.io/api/optimizers/
            # optimizer = hp.Choice('Optimizer', values = ['SGD', 'RMSprop', 'Adam', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl'])
            # Batch sizes
            self.batch_size = hp.Choice('Batch_Size', values = [16, 32, 48, 64])
            # Learning rates
            self.learning_rate = hp.Choice('Learning_Rate', values = [0.1, 0.01, 0.001, 0.0001, 0.00001])


            ## BUILDING THE AUTOENCODER

            # Initialiser function
            initializer = tf.keras.initializers.GlorotNormal(seed = 15)
            # Defining the input
            input = tf.keras.Input(shape = (self.X_DL.shape[1], ), name = 'Input_Layer')
            x = input
            # Defining the encoder structure
            for i in range(self.n_hidden-1):
                x = tf.keras.layers.Dense(self.layers_dims[i], activation = self.activation_function, kernel_initializer = initializer, name = f'Encoder_{i+1}')(x)
            # Defining the last hidden layer -> latent space
            x = tf.keras.layers.Dense(self.layers_dims[-1], activation = self.activation_function, kernel_initializer = initializer, name = 'Encoder_Output')(x)
            # Defining that the encoder output will be equal to the decoder input, that is equal to x for now
            encoder_output = decoder_input = x
            # Defining the decoder structure
            for i in range(len(self.layers_dims)-1, 0, -1):
                x = tf.keras.layers.Dense(self.layers_dims[i], activation = self.activation_function, kernel_initializer = initializer, name = f'Decoder_{len(self.layers_dims)-i}')(x)
            # Defining the last hidden layer -> output
            output = tf.keras.layers.Dense(self.X_DL.shape[1], activation = self.activation_function, kernel_initializer = initializer, name = 'Decoder_Output')(x)

            # Splitting also the encoder and decoder structures
            encoder = tf.keras.Model(input, encoder_output, name = 'Encoder')
            decoder = tf.keras.Model(decoder_input, output, name = 'Decoder')

            # Defining our autoencoder
            autoencoder = tf.keras.Model(input, decoder(encoder(input)), name = 'Autoencoder')

            # Compiling the model
            autoencoder.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate), loss = 'mse', metrics = [tf.keras.metrics.RootMeanSquaredError()])

            return autoencoder
        

        # Instantiating the tuner
        self.tuner = kt.BayesianOptimization(tuning_ae,
                        objective = 'val_loss',
                        max_trials = self.num_trials, 
                        directory = 'AutoML_Experiments',
                        project_name = 'Autoencoder',
                        overwrite = True
                        )

        
        # Generating the train and test splits
        self.train_X, self.test_X, self.train_y, self.test_y = self.pre_processing()

        # Initializing the tuner search - that will basically iterate over a certain number of different combinations (defined in the tuner above)
        self.tuner.search(self.train_X, self.train_X, epochs = self.trial_epochs, batch_size = self.batch_size, validation_split = self.validation_split, callbacks = [self.early_stop])

        print('The best parameters were the following: \n', self.tuner.get_best_hyperparameters(num_trials = 1)[0], '\n')
        # Returns the best model's hyperparameters
        return self.tuner.get_best_hyperparameters(num_trials = 1)




    # Function that conducts scikit-learn hyperparameter optimisation - only to be applied when optimising a Non-DL model
    # Before using this function, the user must define the model to be optimised, as well as its hyperparameters
    def Non_DL_optimise(self):

        """ Method that optimises the hyperparameters of a Non-DL model, using the scikit-optimisation library"""
        
        # Loading the search space for the particular model
        with open(f'{self.search_space_path}/ss_{self.model_name.lower()}.txt', 'rb') as fp:
            self.search_space = pickle.load(fp)

        # Defining the model
        self.define_model()

        # Create the training and testing splits
        self.train_X, self.test_X, self.train_y, self.test_y = self.pre_processing()

        self.n_components_list = []

        # Defining the evaluator function
        @use_named_args(self.search_space)

        def evaluate_model(**params):
            # Defining the models and its params
            self.model.set_params(**params)
            
            if self.model_name == 'RF' or self.model_name == 'LDA':
                self.model.fit(self.train_X, self.train_y)

                if self.model_name == 'RF':
                    # Gathering RF's more important features
                    feature_imp = list(zip(np.arange(self.X_NonDL.shape[1]), self.model.feature_importances_))
                    feature_imp.sort(key = lambda x: x[1], reverse = True)
                    # Defining the number of components to be selected
                    n_components = np.random.randint(2, self.X_NonDL.shape[1])
                    self.n_components_list.append(n_components)
                    # Getting the indexes and the test rf set
                    indexes = [tup[0] for tup in feature_imp[:n_components]]
                    # Reducing the dataset to only contain the features it selected
                    X_transformed = self.train_X[:, indexes]

                else:
                    # Getting the reduced dataset
                    X_transformed = self.model.transform(self.train_X)

            elif self.model_name == 'RFE': 
                feature_selector = self.model.fit(self.train_X, self.train_y)
                # Reducing the dataset to only contain the features it selected
                X_transformed = self.train_X[:, feature_selector.support_]

            else:
                self.model.fit(self.train_X)
                # Getting the reduced dataset
                X_transformed = self.model.transform(self.train_X)
                

            # Define the model to be used for the classification
            log_reg = LogisticRegression()
            # Define the evaluation procedure
            cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 10, random_state = 42)
            # Evaluate the model
            result = cross_val_score(log_reg, X_transformed, self.train_y, cv = cv, n_jobs = -1, scoring = 'accuracy', verbose = 1)
            
            # Return the mean accuracy
            return 1.0 - result.mean()  
            
        self.result = gp_minimize(evaluate_model, self.search_space, n_calls = self.num_trials, n_random_starts = 10, random_state = 42, verbose = 1)

        # Printing the best gotten parameters
        print('The best parameters were the following: \n', self.result.x, '\n')

        # Storing the results
        self.store_results()


    # Method to store all the iterations + results into a csv file, for future analysis
    def store_results(self):

        """ Method that stores the results of the hyperparameter optimisation process into a CSV file"""

        # Getting the columns' names
        cols = []
        for item in self.search_space:
            cols.append(item.name)
        # Apeending also the results columns
        cols.append('Results')
        # Creating an empty dataframe with these columns names
        results_df = pd.DataFrame(columns = cols)

        # Adding the results to the dataframe
        for i, result in enumerate(self.result.x_iters):
            result.append(self.result.func_vals[i])
            results_df.loc[i] = result
        
        if self.model_name == 'RF':
            results_df['n_components'] = self.n_components_list

        # Storing the DataFrame as a CSV file
        if not os.path.exists(self.opt_results):
            os.makedirs(self.opt_results)
        results_df.to_csv(f'{self.opt_results}/{self.model_name}_opt_results.csv', index = False)
        


    # Funcgtion that stores the best model + its hyperparameters
    def store_best_model(self):

        """ Method that stores the best model + its hyperparameters into some files"""

        # Training the best model, in case of a Non_DL model
        if self.model_name not in self.DL_models:
            # Getting the best parameters from the optimisation process
            model_parameters, best_parameters = dict(), dict()
            for key, value in zip(self.search_space, self.result.x):
                best_parameters[key.name] = str(value)
                model_parameters[key.name] = value
            # For the models where there is no "n_components" parameter
            if self.model_name == 'RF':
                best_parameters['n_components'] = str(max(self.n_components_list))
            # Using the best parameters to train the model
            self.model.set_params(**model_parameters)
            if self.model_name != 'RFE' and self.model_name != 'RF' and self.model_name != 'LDA':
                self.model.fit(self.train_X)
            else:
                self.model.fit(self.train_X, self.train_y)

        # Training the best model, in case of a DL model
        else:
            # Retrieving the best hyperparameters from the tuner
            best_hps = self.tuner.get_best_hyperparameters(num_trials = 1)
            # Building the model
            best_model = self.tuner.hypermodel.build(best_hps[0])
            # Compiling the model
            best_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate), loss = 'mse', metrics = [tf.keras.metrics.RootMeanSquaredError()])
            # Training the model
            history = best_model.fit(self.train_X, self.train_X, epochs = self.epochs, batch_size = self.batch_size, validation_split = self.validation_split, callbacks = [self.early_stop, self.cp]).history
            # Taking the encoder
            encoder = tf.keras.Model(best_model.input, best_model.layers[-2].output)
        

        # Saving the best parameters + the model + the weights (in case of DL methods)
        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)
        if not os.path.exists(self.parameters_path):
            os.makedirs(self.parameters_path)

        
        if self.model_name in self.DL_models:
            tf.keras.models.save_model(encoder, f'{self.models_path}/{self.model_name}.h5', save_format = 'h5')
            # encoder.save(f'{self.models_path}/{self.model_name}.h5', save_format = 'h5')
            encoder.save_weights(f'{self.parameters_path}/{self.model_name}_weights.h5', save_format = 'h5')
            with open(f'{self.parameters_path}/{self.model_name}_hyperparameters.json', 'wb') as fp:
                pickle.dump(best_hps[0].values, fp)

        
        else:
            with open(f'{self.parameters_path}/{self.model_name}.json', 'w') as fp:
                json.dump(best_parameters, fp, indent = 3)
            with open(f'{self.models_path}/{self.model_name}.pkl', 'wb') as fp:
                pickle.dump(self.model, fp)

        print('The best model was stored in the following path:', self.models_path, '\n\n')

