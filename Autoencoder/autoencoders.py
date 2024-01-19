#!/usr/bin/env python
# coding: utf-8

# Identical notebook to 'autoencoders.ipynb', but in this one I use the MinMaxScaler instead of StandardScaler for the autoencoder data

# In[1]:


import pandas as pd 
import numpy as np
import tensorflow as tf
import keras_tuner as kt
from keras_tuner import HyperParameters as hp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# ## Step 1 - Loading data

# In[ ]:


X = np.loadtxt('Processed_Files/X_MinMaxScaler.csv', delimiter = ',')
y = np.loadtxt('Processed_Files/y_LabelEncoder.csv', delimiter = ',')


# In[12]:


# Creating a train and test split
globals()['train_X'], test_X, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 42, shuffle = True)


# ## Step 2 - Defining the function that builds the autoencoder + its hyperparameters

# In[13]:


def tune_autoencoder(hp):

    ## DEFINING THE HYPERPARAMETERS TO BE TUNED
    # # Latent space size, i.e., number of reduced dimensions
    # latent_space = hp.Int('Latent_Dimension', min_value = 2, max_value = X.shhape[1])
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


    ## BUILDING THE AUTOENCODER

    # Initialiser function
    initializer = tf.keras.initializers.GlorotNormal(seed = 15)

    # Defining the input
    input = tf.keras.Input(shape = (globals()['train_X'].shape[1], ), name = 'Input_Layer')
    x = input

    # Defining the encoder structure
    for i in range(n_hidden-1):
        x = tf.keras.layers.Dense(layers_dims[i], activation = activation_function, kernel_initializer = initializer, name = f'Encoder_{i+1}')(x)
    # Defining the last hidden layer -> latent space
    x = tf.keras.layers.Dense(layers_dims[-1], activation = activation_function, kernel_initializer = initializer, name = 'Encoder_Output')(x)

    # Defining that the encoder output will be equal to the decoder input, that is equal to x for now
    encoder_output = decoder_input = x

    # Defining the decoder structure
    for i in range(len(layers_dims)-1, 0, -1):
        x = tf.keras.layers.Dense(layers_dims[i], activation = activation_function, kernel_initializer = initializer, name = f'Decoder_{len(layers_dims)-i}')(x)
    # Defining the last hidden layer -> output
    output = tf.keras.layers.Dense(globals()['train_X'].shape[1], activation = activation_function, kernel_initializer = initializer, name = 'Decoder_Output')(x)

    # Splitting also the encoder and decoder structures
    encoder = tf.keras.Model(input, encoder_output, name = 'Encoder')
    decoder = tf.keras.Model(decoder_input, output, name = 'Decoder')

    # Defining our autoencoder
    autoencoder = tf.keras.Model(input, decoder(encoder(input)), name = 'Autoencoder')

    # Compiling the model
    autoencoder.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate), loss = 'mse', metrics = [tf.keras.metrics.RootMeanSquaredError()])

    return autoencoder


# ## Step 3 - Using tuner to tune the autoencoder

# In[14]:


tuner = kt.BayesianOptimization(tune_autoencoder,
                    objective = 'val_loss',
                    max_trials = 20, 
                    directory = 'AutoML_Experiments',
                    project_name = 'Initial_Trial',
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
cp = tf.keras.callbacks.ModelCheckpoint(filepath = 'Best_Model/best_model.h5',
                                mode = 'min', monitor = 'val_loss', verbose = 2 , save_best_only = True)

# Initializing the tuner search - that will basically iterate over a certain number of different combinations (defined in the tuner above)
tuner.search(globals()['train_X'], globals()['train_X'], epochs = 5, batch_size = globals()['batch_size'], validation_split = 0.1, callbacks = [early_stop])

# Printing a summary with the results obtained during the tuning process
tuner.results_summary()


# In[15]:


## RETRIEVING THE BEST MODEL

# Getting the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(best_hps.values)


# In[16]:


# Retrieving the best model
model = tuner.hypermodel.build(best_hps)
model.summary()
# Final fitting of the model
history = model.fit(globals()['train_X'], globals()['train_X'], epochs = 100, batch_size = best_hps.values['Batch_Size'], validation_split = 0.1, callbacks = [early_stop, cp]).history


# In[25]:


# Retrieving the encoder model - what actually matters for Dimensionality Reduction
encoder = tf.keras.Model(model.input, model.layers[-2].output)
encoder.summary()


# ## Step 4 - Proceeding with Dimensionality Reduction study and comparison

# In[26]:


# Getting the encoder reduced data 
encoder_reduced_train = encoder.predict(globals()['train_X'])
encoder_reduced_test = encoder.predict(test_X)


# In[27]:


# Importing all the classifiers to be used
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

# Importing metrics
from sklearn.metrics import f1_score


# In[28]:


# Loading each classifier (with their default hyperparameters)
svm = SVC()
rf = RandomForestClassifier()
gb = GradientBoostingClassifier()
nb = GaussianNB()
log_reg = LogisticRegression()


# In[29]:


# Fitting and applying all classifiers to the original sized dataset
svm.fit(globals()['train_X'], train_y)
rf.fit(globals()['train_X'], train_y)
gb.fit(globals()['train_X'], train_y)
nb.fit(globals()['train_X'], train_y)
log_reg.fit(globals()['train_X'], train_y)

# Predicting the results of the test set
y_pred_svm = svm.predict(test_X)
y_pred_rf = rf.predict(test_X)
y_pred_gb = gb.predict(test_X)
y_pred_nb = nb.predict(test_X)
y_pred_log_reg = log_reg.predict(test_X)

# Calculating the metrics for each classifier
f1_svm = f1_score(test_y, y_pred_svm, average = 'weighted')
f1_rf = f1_score(test_y, y_pred_rf, average = 'weighted')
f1_gb = f1_score(test_y, y_pred_gb, average = 'weighted')
f1_nb = f1_score(test_y, y_pred_nb, average = 'weighted')
f1_log_reg = f1_score(test_y, y_pred_log_reg, average = 'weighted')

# Storing the metrics under a dataframe
metrics = pd.DataFrame(columns = ['N_vars', 'SVM', 'RF', 'GB', 'NB', 'LogReg'])
metrics.loc[0] = ['All', f1_svm, f1_rf, f1_gb, f1_nb, f1_log_reg]


# In[30]:


# Fitting and applying all classifiers to the reduced dataset
svm.fit(encoder_reduced_train, train_y)
rf.fit(encoder_reduced_train, train_y)
gb.fit(encoder_reduced_train, train_y)
nb.fit(encoder_reduced_train, train_y)
log_reg.fit(encoder_reduced_train, train_y)

# Predicting the results of the test set
y_pred_svm_encoder = svm.predict(encoder_reduced_test)
y_pred_rf_encoder = rf.predict(encoder_reduced_test)
y_pred_gb_encoder = gb.predict(encoder_reduced_test)
y_pred_nb_encoder = nb.predict(encoder_reduced_test)
y_pred_log_reg_encoder = log_reg.predict(encoder_reduced_test)

# Calculating the metrics for each classifier
f1_svm_encoder = f1_score(test_y, y_pred_svm_encoder, average = 'weighted')
f1_rf_encoder = f1_score(test_y, y_pred_rf_encoder, average = 'weighted')
f1_gb_encoder = f1_score(test_y, y_pred_gb_encoder, average = 'weighted')
f1_nb_encoder = f1_score(test_y, y_pred_nb_encoder, average = 'weighted')
f1_log_reg_encoder = f1_score(test_y, y_pred_log_reg_encoder, average = 'weighted')

# Storing the metrics under a dataframe
metrics.loc[1] = [encoder.output.shape[1], f1_svm_encoder, f1_rf_encoder, f1_gb_encoder, f1_nb_encoder, f1_log_reg_encoder]


# In[31]:


print(metrics)

