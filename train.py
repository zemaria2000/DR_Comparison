from Classes import Optimise
import yaml, sys

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

        # Instantiating the class
        model_optimiser = Optimise.hyperparameter_optimiser(model_name)

        # Promoting the model optimisation
        if model_name in DL_models:
            if model_name == 'VAE':
                model_optimiser.build_optimised_vae()
            if model_name == 'AE':
                model_optimiser.build_optimised_autoencoder()
        else:
            model_optimiser.Non_DL_optimise()

        # Store the best model + hyperparameters
        model_optimiser.store_best_model()


else:

    # Analysing the model that was given:
    if len(args) == 2 and args[1].upper() not in available_models:
        raise SyntaxError(f"The model given is not available... Please provide a model to be trained from the following list: {Configs['models']['available_models']}")
    if len(args) > 2:
        raise SyntaxError("Too many arguments were given... Please provide only one model to be trained, or don't provide any if you want to train all the models")
    else: 
        model_name = args[1].upper()

    # Instantiate the optimiser class
    model_optimiser = Optimise.hyperparameter_optimiser(model_name)

    # Promoting the model optimisation
    if model_name in DL_models:
        if model_name == 'VAE':
            model_optimiser.build_optimised_vae()
        if model_name == 'AE':
            model_optimiser.build_optimised_autoencoder()
    else:
        model_optimiser.Non_DL_optimise()

    # Store the best model + hyperparameters
    model_optimiser.store_best_model()

