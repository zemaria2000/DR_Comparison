from Classes import Optimizer, Autoencoders
import yaml, sys
# Importing the models
from sklearn.decomposition import PCA, FastICA
import pickle
from keras_tuner import HyperParameters as hp


# We should launch the file as "python train.py <model>, the model being a string correspondant to the model to be trained
# Current possible strings to be used - ['PCA', 'FastICA', 'AE', 'VAE']
args = sys.argv
# Loading the settings file
with open('settings.yaml') as file:
    Configs = yaml.full_load(file)

DL_models = Configs['models']['DL_models']

# Analysing the model that was given:
if len(args) == 2 and args[1] not in Configs['models']['available_models']:
    print(f"The model given is not available... Please provide a model to be trained from the following list: {Configs['models']['available_models']}")
    sys.exit()
if len(args) > 2:
    print('Too many arguments were given... Please provide only one model to be trained')
    sys.exit()
if len(args) < 1:
    print(f"No model was given... Please provide a model to be trained with the follwing exact syntax: {Configs['models']['available_models']}")
    sys.exit()
else: 
    if args[1] == 'PCA':
        model = PCA()
    if args[1] == 'FastICA':
        model = FastICA()
    else:
        model = args[1]


## CONDUCTING THE OPTIMIZATION PROCESS FOR NON-DL MODELS
        
if args[1] not in DL_models:

    # Instantiating the classes
    optimizer = Optimizer.ModelOptimisation(model, Configs['models']['num_trials'], Configs['models']['train_split'], Configs['models']['validation_split'])

    # ## DATA PRE-PROCESSING
    # train_X, test_X, train_y, test_y = optimizer.pre_processing()

    ## OPTIMISATION PROCESS

    # Defining the search space
    if args[1] == 'PCA':
        with open(f'./SearchSpace/ss_pca.txt', 'rb') as fp:
            search_space = pickle.load(fp)
    if args[1] == 'FastICA':    
        with open(f'./SearchSpace/ss_ica.txt', 'rb') as fp:
            search_space = pickle.load(fp)
    # conducting the hyperparameter optimisation
    result = optimizer.Non_DL_optimise(search_space)

    # Storing the best model + parameters
    optimizer.store_best_model(search_space, result, args[1])


## CONDUCTING THE OPTIMIZATION PROCESS FOR DL MODELS
    
else:

    # Instantiating the class
    optimizer = Autoencoders.Autoencoders(Configs['models']['num_trials'], Configs['models']['train_split'], Configs['models']['validation_split'])


    ## Optimization process

    # Choosing the correct model and conducting the search
    if model == 'AE':
        best_model = optimizer.build_autoencoder()
    if model == 'VAE':
        best_model = optimizer.build_vae()

    # Saving the best model
    optimizer.save_model(best_model, model)