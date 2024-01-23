
import pandas as pd 
import numpy as np 
import tensorflow as tf 
import keras_tuner as kt 
from keras_tuner import HyperParameters as hp 
from sklearn.model_selection import train_test_split
import os
import yaml

## Loading some important variables
with open('../settings.yaml') as file:
    Configs = yaml.full_load(file)

# Defining the path to the data
data_path, model_path = Configs['directories']['datasets'], Configs['directories']['best_model']
# Getting some important variables
num_trials, train_split, validation_split, trial_epochs, epochs = Configs['models']['num_trials'], Configs['models']['train_split'], Configs['models']['validation_split'], Configs['models']['num_epochs_trials'], Configs['models']['num_final_epochs']


# ## Step 1 - Loading Data
X = np.loadtxt(f'{data_path}/X_MinMaxScaler.csv', delimiter = ',')
y = np.loadtxt(f'{data_path}/y_LabelEncoder.csv', delimiter = ',')

# Creating a train and test split
globals()['train_X'], test_X, train_y, test_y = train_test_split(X, y, test_size = 1-train_split, random_state = 42, shuffle = True)


# ## Step 2 - Defining the function that builds the VAE + its hyperparameters

from tensorflow.keras import backend as K

def tune_VAE(hp):

    ## DEFINING THE HYPERPARAMETERS TO BE TUNED

    # Number of hiddewn layers
    n_hidden = hp.Int('Hidden_Layers', min_value = 3, max_value = 7)
    # Drop between each layer, which will define the size of the subsequent layer
    layers_drop = []
    for i in range(n_hidden):
        layers_drop.append(hp.Float(f"drop_{i}-{i+1}", min_value = 1.2, max_value = 1.8))
    # Layer dimensions, which depend on drop between layers
    layers_dims = []
    for i in range(n_hidden):
        if i == 0:      # first layer
            layers_dims.append(int(globals()['train_X'].shape[1]/layers_drop[i]))
        else:
            layers_dims.append(int(layers_dims[i-1]/layers_drop[i]))
    # Activation function - https://keras.io/2.15/api/layers/activations/
    activation_function = hp.Choice('Activation_Function', values = ['relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh', 'selu', 'elu'])
    # # Optimizer - https://keras.io/api/optimizers/
    # optimizer = hp.Choice('Optimizer', values = ['SGD', 'RMSprop', 'Adam', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl'])
    # Batch sizes
    globals()['batch_size'] = hp.Choice('Batch_Size', values = [16, 32, 48, 64])
    # batch_size = hp.Choice('Batch_Size', values = [16, 32, 48, 64])
    # Learning rates
    learning_rate = hp.Choice('Learning_Rate', values = [0.1, 0.01, 0.001, 0.0001, 0.00001])
    # # Cost function weight
    # weight = hp.Float('Reconstruction_Weight', min_value = 0.1, max_value = 0.9)


    ## DEFINE THE SAMPLING FUNCTION FOR THE LATENT SPACE SAMPLE GENERATION
    def sampling(args):
        z_mean, z_log_sigma, latent_dim = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
        return z_mean + K.exp(z_log_sigma) * epsilon


    ## BUILDING THE VAE MODEL

    # Initialiser function
    initializer = tf.keras.initializers.GlorotNormal(seed = 15)

    # Defining the input
    input = tf.keras.Input(shape = (globals()['train_X'].shape[1], ), name = 'Input_Layer')
    x = input

    # Defining the encoder structure
    for i in range(n_hidden-1):
        x = tf.keras.layers.Dense(layers_dims[i], activation = activation_function, kernel_initializer = initializer, name = f'Encoder_{i+1}')(x)
    # Defining the last hidden layer -> latent space
    z_mean = tf.keras.layers.Dense(layers_dims[-1], name = 'Z_mean')(x)
    z_log_sigma = tf.keras.layers.Dense(layers_dims[-1], name = 'Z_Log_Sigma')(x)
    z = tf.keras.layers.Lambda(sampling, name = 'Z_Sampling_Layer')([z_mean, z_log_sigma, layers_dims[-1]])

    # Building the decoder
    latent_inputs = tf.keras.Input(shape = (layers_dims[-1], ), name = 'Input_Z_Sampling')
    x = latent_inputs
    # Decoder layers
    for i in range(len(layers_dims)-1, 0, -1):
        x = tf.keras.layers.Dense(layers_dims[i], activation = activation_function, kernel_initializer = initializer, name = f'Decoder_{len(layers_dims)-i}')(x)
    # Defining the last hidden layer -> output
    output = tf.keras.layers.Dense(globals()['train_X'].shape[1], activation = activation_function, kernel_initializer = initializer, name = 'Decoder_Output')(x)

    # # Splitting also the encoder and decoder structures
    encoder = tf.keras.Model(input, [z_mean, z_log_sigma, z], name = 'Encoder')
    decoder = tf.keras.Model(latent_inputs, output, name = 'Decoder')

    # Defining our VAE
    output_vae = decoder(encoder(input)[2])
    vae = tf.keras.Model(input, output_vae, name = 'VAE')

    # Calculating the losses
    reconstruction = layers_dims[0] * tf.keras.losses.mse(input, output_vae)
    kl = -0.5 * K.sum(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis = 1)

    # Total loss function
    # vae_loss = reconstruction*weight + kl*(1 - weight)
    vae_loss = K.mean(reconstruction + kl)    
    vae.add_loss(vae_loss)

    # Compiling the model
    vae.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate))

    return vae


# ## Step 3 - Using tuner to tune the VAE

tuner = kt.BayesianOptimization(tune_VAE,
                    objective = 'val_loss',
                    max_trials = num_trials, 
                    directory = 'AutoML_Experiments',
                    project_name = 'VAEs',
                    overwrite = True
                    )

# Defining a callback that stops the search if the results aren't improving
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_loss',
    min_delta = 0.0001,
    patience = 20,
    verbose = 1, 
    mode = 'min',
    restore_best_weights = True)
# Defining a callback that saves our model
cp = tf.keras.callbacks.ModelCheckpoint(filepath = 'Best_Model/VAE.h5',
                                mode = 'min', monitor = 'val_loss', verbose = 2 , save_best_only = True)

# Initializing the tuner search - that will basically iterate over a certain number of different combinations (defined in the tuner above)
tuner.search(globals()['train_X'], globals()['train_X'], epochs = trial_epochs, batch_size = globals()['batch_size'], validation_split = validation_split, callbacks = [early_stop])

# Printing a summary with the results obtained during the tuning process
tuner.results_summary()


# ## Step 4 - Storing the best vae encoders

# Getting the models with the best hyperparameters
best_models = tuner.get_best_hyperparameters(num_trials=5)

# Training each model and storing it in a particular directory
for element in best_models:
    # Building the model with its optimised hyperparameters
    model = tuner.hypermodel.build(element)
    model.summary()
    # Compiling the model
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = globals()['learning_rate']), loss = 'mse', metrics = [tf.keras.metrics.RootMeanSquaredError()])
    # Training the model
    history = model.fit(globals()['train_X'], globals()['train_X'], epochs = 1000, batch_size = globals()['batch_size'], validation_split = validation_split, callbacks = [early_stop, cp]).history
    # Taking the encoder
    encoder = tf.keras.Model(model.input, model.layers[-2].output)
    if not os.path.exists('../Models/Autoencoders'):
        os.makedirs('../Models/Autoencoders')
        # Saving the model on a specific directory
        encoder.save(f'{model_path}/VAEs/VAE_{best_models.index(element)+1}', save_format = 'h5')
    else:
        encoder.save(f'{model_path}/VAEs/VAE_{best_models.index(element)+1}', save_format = 'h5')
