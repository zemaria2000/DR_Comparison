from Classes import Optimizer, Autoencoders
import yaml, sys
import pickle


# We should launch the file as "python train.py <model>, the model being a string correspondant to the model to be trained
# Current possible strings to be used - ['PCA', 'FastICA', 'AE', 'VAE']
args = sys.argv
# Loading the settings file
with open('settings.yaml') as file:
    Configs = yaml.full_load(file)

# Just loading some important settings
DL_models = Configs['models']['DL_models']
available_models = Configs['models']['available_models']
num_trials, train_split, validation_split = Configs['models']['num_trials'], Configs['models']['train_split'], Configs['models']['validation_split']

# In case we want to train all the models
if len(args) == 1:

    for item in Configs['models']['available_models']:

        print('Conducting hyperparameter optimisation for model', item, '\n')
        # Getting the model to be optimised (in a string format)
        model_name = item

        if item not in DL_models:

            # Instantiating the classe
            optimizer = Optimizer.ModelOptimisation(model_name, num_trials, train_split, validation_split)

            # Loading the search space for the particular model
            with open(f'./SearchSpace/ss_{model_name.lower()}.txt', 'rb') as fp:
                search_space = pickle.load(fp)

            # conducting the hyperparameter optimisation
            result = optimizer.Non_DL_optimise(search_space)
            print("The model's best hyperparameters were the following: \n", result.x, '\n')

            # Storing the best model + parameters
            optimizer.store_best_model(search_space, result)

        else:

            # Instantiating the class
            optimizer = Autoencoders.Autoencoders(num_trials, train_split, validation_split)

            # Choosing the correct model and conducting the search
            if model_name == 'AE':
                best_model = optimizer.build_autoencoder()
            if model_name == 'VAE':
                best_model = optimizer.build_vae()

            # Saving the best model
            optimizer.save_model(best_model, model_name)

        print('Optimisation process concluded for', item, 'model \n\n')

# In case we want to train only one particular model
else:
    
    # Analysing the model that was given:
    if len(args) == 2 and args[1].upper() not in available_models:
        raise SyntaxError(f"The model given is not available... Please provide a model to be trained from the following list: {Configs['models']['available_models']}")
    if len(args) > 2:
        raise SyntaxError("Too many arguments were given... Please provide only one model to be trained, or don't provide any if you want to train all the models")
    else: 
        model_name = args[1].upper()
        
    # If the model is not a DL model, we'll use the ModelOptimisation class
    if model_name not in DL_models:

        # Instantiating the classes
        optimizer = Optimizer.ModelOptimisation(model_name, num_trials, train_split, validation_split)

        # Loading the search space for the particular model
        with open(f'./SearchSpace/ss_{model_name.lower()}.txt', 'rb') as fp:
            search_space = pickle.load(fp)

        # conducting the hyperparameter optimisation
        result = optimizer.Non_DL_optimise(search_space)

        # Storing the best model + parameters
        optimizer.store_best_model(search_space, result)


    # If the model is a DL model, we'll use the Autoencoders class
    else:

        # Instantiating the class
        optimizer = Autoencoders.Autoencoders(num_trials, train_split, validation_split)

        # Choosing the correct model and conducting the search
        if model_name == 'AE':
            best_model = optimizer.build_autoencoder()
        if model_name == 'VAE':
            best_model = optimizer.build_vae()

        # Saving the best model
        optimizer.save_model(best_model, model_name)