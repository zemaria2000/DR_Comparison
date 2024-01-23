import tensorflow as tf 
import keras_tuner as kt
from keras_tuner import HyperParameters as hp
import numpy as np
from tensorflow.keras import backend as K
import os, yaml
from sklearn.model_selection import train_test_split

class Autoencoders:

    def __init__(self, num_trials, train_split, validation_split):
        
        with open('settings.yaml') as file:
            Configs = yaml.full_load(file)

        self.dataset_path = Configs['directories']['datasets']
        self.model_path = Configs['directories']['best_model']
        self.num_trials = num_trials
        self.train_split = train_split
        self.validation_split = validation_split
        self.X = np.loadtxt(f'{self.dataset_path}/X_MinMaxScaler.csv', delimiter = ',')
        self.y = np.loadtxt(f'{self.dataset_path}/y_LabelEncoder.csv', delimiter = ',')
        self.trial_epochs = Configs['models']['num_epochs_trials']
        self.epochs = Configs['models']['num_final_epochs']

        # Defining a callback that saves our model
        self.cp = tf.keras.callbacks.ModelCheckpoint(filepath = 'Best_Model/VAE.h5',
                                        mode = 'min', monitor = 'val_loss', verbose = 2 , save_best_only = True)
        # Defining a callback that stops the search if the results aren't improving
        self.early_stop = tf.keras.callbacks.EarlyStopping(
            monitor = 'val_loss',
            min_delta = 0.0001,
            patience = 20,
            verbose = 1, 
            mode = 'min',
            restore_best_weights = True)
        

    # Creating the training and testing splits
    def pre_processing(self):
        # Returning the train and test splits
        return train_test_split(self.X, self.y, test_size = 1-self.train_split, random_state = 42, shuffle = True)
    
    # Function that builds an autoencoder
    def build_autoencoder(self):

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
                    self.layers_dims.append(int(self.X.shape[1]/self.layers_drop[i]))
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
            input = tf.keras.Input(shape = (self.X.shape[1], ), name = 'Input_Layer')
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
            output = tf.keras.layers.Dense(self.X.shape[1], activation = self.activation_function, kernel_initializer = initializer, name = 'Decoder_Output')(x)

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

        # Returns the best model's hyperparameters
        return self.tuner.get_best_hyperparameters(num_trials = 1)


    # Function that builds a VAE
    def build_vae(self):   

        def tuning_vae(hp):

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
                    self.layers_dims.append(int(self.X.shape[1]/self.layers_drop[i]))
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
 

            ## DEFINE THE SAMPLING FUNCTION FOR THE LATENT SPACE SAMPLE GENERATION
            def sampling(args):
                z_mean, z_log_sigma, latent_dim = args
                epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
                return z_mean + K.exp(z_log_sigma) * epsilon

            ## BUILDING THE VAE MODEL

            # Initialiser function
            initializer = tf.keras.initializers.GlorotNormal(seed = 15)

            # Defining the input
            input = tf.keras.Input(shape = (self.X.shape[1], ), name = 'Input_Layer')
            x = input

            # Defining the encoder structure
            for i in range(self.n_hidden-1):
                x = tf.keras.layers.Dense(self.layers_dims[i], activation = self.activation_function, kernel_initializer = initializer, name = f'Encoder_{i+1}')(x)
            # Defining the last hidden layer -> latent space
            z_mean = tf.keras.layers.Dense(self.layers_dims[-1], name = 'Z_mean')(x)
            z_log_sigma = tf.keras.layers.Dense(self.layers_dims[-1], name = 'Z_Log_Sigma')(x)
            z = tf.keras.layers.Lambda(sampling, name = 'Z_Sampling_Layer')([z_mean, z_log_sigma, self.layers_dims[-1]])

            # Building the decoder
            latent_inputs = tf.keras.Input(shape = (self.layers_dims[-1], ), name = 'Input_Z_Sampling')
            x = latent_inputs
            # Decoder layers
            for i in range(len(self.layers_dims)-1, 0, -1):
                x = tf.keras.layers.Dense(self.layers_dims[i], activation = self.activation_function, kernel_initializer = initializer, name = f'Decoder_{len(self.layers_dims)-i}')(x)
            # Defining the last hidden layer -> output
            output = tf.keras.layers.Dense(self.X.shape[1], activation = self.activation_function, kernel_initializer = initializer, name = 'Decoder_Output')(x)

            # # Splitting also the encoder and decoder structures
            encoder = tf.keras.Model(input, [z_mean, z_log_sigma, z], name = 'Encoder')
            decoder = tf.keras.Model(latent_inputs, output, name = 'Decoder')

            # Defining our VAE
            output_vae = decoder(encoder(input)[2])
            vae = tf.keras.Model(input, output_vae, name = 'VAE')

            # Calculating the losses
            reconstruction = self.layers_dims[0] * tf.keras.losses.mse(input, output_vae)
            kl = -0.5 * K.sum(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis = 1)

            # Total loss function
            # vae_loss = reconstruction*weight + kl*(1 - weight)
            vae_loss = K.mean(reconstruction + kl)    
            vae.add_loss(vae_loss)

            # Compiling the model
            vae.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate))

            return vae
    
        # Instantiating the tuner
        self.tuner = kt.BayesianOptimization(tuning_vae,
                        objective = 'val_loss',
                        max_trials = self.num_trials, 
                        directory = 'AutoML_Experiments',
                        project_name = 'VAE',
                        overwrite = True
                        )
                
        # Generating the train and test splits
        self.train_X, self.test_X, self.train_y, self.test_y = self.pre_processing()

        # Initializing the tuner search - that will basically iterate over a certain number of different combinations (defined in the tuner above)
        self.tuner.search(self.train_X, self.train_X, epochs = self.trial_epochs, batch_size = self.batch_size, validation_split = self.validation_split, callbacks = [self.early_stop])

        # Returns the best model's hyperparameters
        return self.tuner.get_best_hyperparameters(num_trials = 1)




    # def hyperparameter_tuner(self, model: str):

    #     if model == 'AE':
    #         # Instantiating the tuner
    #         self.tuner = kt.BayesianOptimization(self.build_autoencoder,
    #                         objective = 'val_loss',
    #                         max_trials = self.num_trials, 
    #                         directory = 'AutoML_Experiments',
    #                         project_name = 'Autoencoder',
    #                         overwrite = True
    #                         )
            
    #         # Defining a callback that saves our model
    #         self.cp = tf.keras.callbacks.ModelCheckpoint(filepath = 'Best_Model/AE.h5',
    #                                         mode = 'min', monitor = 'val_loss', verbose = 2 , save_best_only = True)
            

    #     if model == 'VAE':
    #         # Instantiating the tuner
    #         self.tuner = kt.BayesianOptimization(self.build_vae,
    #                         objective = 'val_loss',
    #                         max_trials = self.num_trials, 
    #                         directory = 'AutoML_Experiments',
    #                         project_name = 'VAE',
    #                         overwrite = True
    #                         )
            
    #         # Defining a callback that saves our model
    #         self.cp = tf.keras.callbacks.ModelCheckpoint(filepath = 'Best_Model/VAE.h5',
    #                                         mode = 'min', monitor = 'val_loss', verbose = 2 , save_best_only = True)
            
        
    #     # Defining a callback that stops the search if the results aren't improving
    #     self.early_stop = tf.keras.callbacks.EarlyStopping(
    #         monitor = 'val_loss',
    #         min_delta = 0.0001,
    #         patience = 20,
    #         verbose = 1, 
    #         mode = 'min',
    #         restore_best_weights = True)
        
    #     # Generating the train and test splits
    #     self.train_X, self.test_X, self.train_y, self.test_y = self.pre_processing(self)

    #     # Initializing the tuner search - that will basically iterate over a certain number of different combinations (defined in the tuner above)
    #     self.tuner.search(self.train_X, self.train_X, epochs = self.trial_epochs, batch_size = self.batch_size, validation_split = self.validation_split, callbacks = [self.early_stop])

    #     # Returns the best model's hyperparameters
    #     return self.tuner.get_best_hyperparameters(num_trials = 1)
    

    def save_model(self, best_hps, model: str):

        # Building the model that yielded the best hyperparameters
        best_model = self.tuner.hypermodel.build(best_hps[0])
        best_model.summary()
        # Compiling the model
        best_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate), loss = 'mse', metrics = [tf.keras.metrics.RootMeanSquaredError()])
         # Training the model
        history = best_model.fit(self.train_X, self.train_X, epochs = self.epochs, batch_size = self.batch_size, validation_split = self.validation_split, callbacks = [self.early_stop, self.cp]).history
        # Taking the encoder
        encoder = tf.keras.Model(best_model.input, best_model.layers[-2].output)

        # Saving the model on a specific directory
        if not os.path.exists(f'{self.model_path}'):
            os.makedirs(f'{self.model_path}')
            encoder.save(f'{self.model_path}/{model}.h5', save_format = 'h5')
        else:
            encoder.save(f'{self.model_path}/{model}.h5', save_format = 'h5')
