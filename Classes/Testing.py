""" This class is used to test the models we have trained.
It allows us to test the models with the full dataset and with the reduced datasets. 
It also allows us to retrieve some metrics from the results. """

import numpy as np 
import pandas as pd
import yaml, pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, multilabel_confusion_matrix, matthews_corrcoef
from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.decomposition import TruncatedSVD as SVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier as DT
import tensorflow as tf
from tensorflow.keras import backend as K


class Testing:

    def __init__(self, model_name):

        with open('settings.yaml') as file:
            Configs = yaml.full_load(file)

        self.model_name = model_name
        self.dataset_path = Configs['directories']['datasets']
        self.model_path = Configs['directories']['best_model']
        self.parameters_path = Configs['directories']['parameters']
        self.train_split = Configs['models']['train_split']
        self.validation_split = Configs['models']['validation_split']
        self.DL_models = Configs['models']['DL_models']
        self.X_DL = np.loadtxt(f'{self.dataset_path}/X_MinMaxScaler.csv', delimiter = ',')
        self.X = np.loadtxt(f'{self.dataset_path}/X_StandardScaler.csv', delimiter = ',')
        self.y = np.loadtxt(f'{self.dataset_path}/y_LabelEncoder.csv', delimiter = ',')


    # Data Preprocessing    
    def pre_processing(self):

        """ This method is used to preprocess the data before training the models. It returns the training and testing sets
        that are appropriate for the model to be trained"""

        if self.model_name not in self.DL_models and self.model_name != 'NMF':
            return train_test_split(self.X, self.y, test_size = 1-self.train_split, random_state = 42, shuffle = True)
        else:
            return train_test_split(self.X_DL, self.y, test_size = 1-self.train_split, random_state = 42, shuffle = True)


    # Defining the model to be used
    def load_default_model(self):
        
        """ Method that defines the model to be trained, according to the model's name"""

        if self.model_name == 'PCA':
            default_model = PCA()
        elif self.model_name == 'ICA':
            default_model = FastICA(max_iter = 5000, tol = 5e-2)
        elif self.model_name == 'NMF':
            default_model = NMF(max_iter = 10000, tol = 5e-2)
        elif self.model_name == 'SVD':
            default_model = SVD()
        elif self.model_name == 'LDA':
            default_model = LDA(tol = 5e-2)
        elif self.model_name == 'RF':
            default_model = RF()
        elif self.model_name == 'RFE':
            default_model = RFECV(estimator = DT())
        elif self.model_name == 'AE':
            default_model = self.build_ae([self.X.shape[1], 32, 16, 8, 4])
        
        return default_model
            
    # Loading the best model
    def load_best_model(self):

        """ Method that loads the best model from the optimisation process """

        if self.model_name not in self.DL_models:
            with open(f'{self.model_path}/{self.model_name}.pkl', 'rb') as fp:
                best_model = pickle.load(fp)
        if self.model_name == 'AE':
            best_model = tf.keras.models.load_model(f'{self.model_path}/{self.model_name}.h5')

        return best_model
        

    # Function that builds a DL "default" model - Autoencoder or Variational Autoencoder
    def build_ae(self, dims: list):

        """ Method that helps building a base autoencoder model, with the given
        dimensions """

        # Defining the initialiser
        initialiser = tf.keras.initializers.GlorotUniform(seed = 15)

        # Defining the number of layers
        n_layers = len(dims) - 1    # The 1st dim value is the input dim

        # Defining our input
        input_layer = tf.keras.Input(shape = (dims[0], ), name = 'Input')
        x = input_layer

        # Defining our encoder HIDDEN layers
        for i in range(n_layers - 1):   # The last layer is not hidden - it corresponds to the latent space
            if i == 0 or i == 1:
                x = tf.keras.layers.Dense(dims[i + 1], activation = 'swish', kernel_initializer = initialiser, name = f'encoder_{i}')(x)
                # x = tf.keras.layers.Dropout(0.2, name = f'dropout_{i+1}')(x)
            else:
                x = tf.keras.layers.Dense(dims[i + 1], activation = 'swish', kernel_initializer = initialiser, name = f'encoder_{i}')(x)
        # Defining our encoder output - latent space
        x = tf.keras.layers.Dense(dims[-1], activation = 'swish', kernel_initializer = initialiser, name = 'encoder_output')(x)

        # Decoder input layer
        # decoder_input = tf.keras.Input(shape = (dims[-1], ), name = 'decoder_input')
        encoder_output = decoder_input = x
        # Defining our decoder
        for i in range(n_layers, 0, -1):
            x = tf.keras.layers.Dense(dims[i], activation = 'swish', kernel_initializer = initialiser, name = f'decoder_{i}')(x)
        # Defining our decoder output
        output = tf.keras.layers.Dense(dims[0], activation = 'swish', name = 'output')(x)

        # Defining our encoder and decoder
        encoder = tf.keras.Model(input_layer, encoder_output, name = 'encoder')
        decoder = tf.keras.Model(decoder_input, output, name = 'decoder')

        # Defining our autoencoder
        autoencoder = tf.keras.Model(input_layer, decoder(encoder(input_layer)), name = 'autoencoder')

        # Compiling the model
        autoencoder.compile(optimizer = 'adam', loss = 'mse')

        return autoencoder


    def fit_DR_technique(self, model, n_components):

        """ Method that fits the model with the training data, and returns the fitted model """

        # Create the training and testing splits
        train_X, test_X, train_y, test_y = self.pre_processing()

        # Setting the models' number of components, when possible
        try:
            model.set_params(n_components = n_components)
        except:
            pass

        if self.model_name in self.DL_models:
            model.fit(train_X, train_X, epochs = 100, validation_split = self.validation_split, batch_size = 64)

        else:
            if self.model_name == 'RF' or self.model_name == 'RFE' or self.model_name == 'LDA':
                model.fit(train_X, train_y)
            else:
                model.fit(train_X)
            
        return model


    def generate_reduced_datasets_nonDL(self, model, n_components):
        
        """ Method that allows us to generate the reduced dataset for non_DL models"""

        # Creating the training and testing splits
        train_X, test_X, train_y, test_y = self.pre_processing()

        # if self.model_name not in self.DL_models:

        if self.model_name != 'RF' and self.model_name != 'RFE':

            # Generating the train and test datasets with the reduced data
            train_X_reduced = model.transform(train_X)
            test_X_reduced = model.transform(test_X)

        if self.model_name == 'RF':

            # Gathering RF's more important features
            feature_imp = list(zip(np.arange(train_X.shape[1]), model.feature_importances_))
            feature_imp.sort(key = lambda x: x[1], reverse = True)
            # Getting the indexes and the test rf set
            indexes = [tup[0] for tup in feature_imp[:n_components]]
            # Reducing the dataset to only contain the features it selected
            train_X_reduced, test_X_reduced = train_X[:, indexes], test_X[:, indexes]
            print(indexes)
            print('\n\n\n\n')

        if self.model_name == 'RFE':

            feature_selector = model.fit(train_X, train_y)
            # Reducing the dataset to only contain the features it selected
            train_X_reduced, test_X_reduced = train_X[:, feature_selector.support_], test_X[:, feature_selector.support_]
            
        return train_X_reduced, test_X_reduced, train_y, test_y


    # method that allows us to generate the reduced dataset for DL models
    def generate_reduced_datasets_DL(self, model):
        
        """ Method that allows us to generate the reduced dataset for DL models"""

        # Creating the training and testing splits
        train_X, test_X, train_y, test_y = self.pre_processing()

        train_X_reduced = model.predict(train_X)
        test_X_reduced = model.predict(test_X)

        return train_X_reduced, test_X_reduced, train_y, test_y
        


    def get_metrics(self, y_true, y_pred, n_components):

        """ Method that allows us to retrieve some metrics from the results """

        mcc = matthews_corrcoef(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average = 'weighted')
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average = 'weighted')
        recall = recall_score(y_true, y_pred, average = 'weighted')

        aux_series = pd.Series([self.model_name, n_components, mcc, f1, acc, prec, recall], index = ['Model', 'N_components', 'MCC', 'F1', 'Accuracy', 'Precision', 'Recall'])
        
        return aux_series