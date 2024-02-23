
""" Script only used as a way to better organise the results for the elaboration of 
the paper. It is not relevant nor significant for the project itself"""

# %%
import pandas as pd
import yaml
# %%

with open('settings.yaml') as file:
    Configs = yaml.full_load(file)
default_tests_path = Configs['directories']['default_tests']
time_tests_path = Configs['directories']['time_tests']
dr_time_tests_path = Configs['directories']['dr_time_tests']
dr_techniques = Configs['models']['available_models']

classifiers_list = ['LogisticRegression', 'KNeighborsClassifier', 'MLPClassifier', 'RandomForestClassifier']
short_classifiers_list = ['LogReg', 'kNN', 'MLP', 'RF']



# %%

# Creating a dataframe for each DR technique for the values of MCC
for dr_technique in dr_techniques:

    df = pd.DataFrame()

    for long_classifier, short_classifier in zip(classifiers_list, short_classifiers_list):

        aux_df = pd.read_csv(f'{default_tests_path}/{dr_technique}_{long_classifier}.csv', index_col = 'Unnamed: 0')
        aux_df = aux_df[['N_components', 'MCC']]
        aux_df.reset_index(drop = True, inplace = True)
        aux_df.rename(columns = {'MCC': f'{short_classifier}'}, inplace = True)
        aux_df[short_classifier] = aux_df[short_classifier].round(3)

        if df.empty:
            df = aux_df
        else:                
            df = pd.concat([df, aux_df[short_classifier]], axis = 1)
    
    df.to_csv(f'{default_tests_path}/{dr_technique}_MCC.csv')


# %%
# Creating a dataframe for each DR technique with the respective classifiers times (training and prediction)
for dr_technique in dr_techniques:

    train_df, test_df = pd.DataFrame(), pd.DataFrame()

    
    aux_df = pd.read_csv(f'{time_tests_path}/{dr_technique}_Times.csv')
    grouped = aux_df.groupby('Classifier')

    for classifier, group in grouped:
        
        group = group[['N_components', 'Training time', 'Testing time']]
        group.reset_index(drop = True, inplace = True)

        if train_df.empty:
            train_df = group[['N_components', 'Training time']]
            train_df.rename(columns = {'Training time': f'{classifier}'}, inplace = True)
        if test_df.empty:
            test_df = group[['N_components', 'Testing time']]
            test_df.rename(columns = {'Training time': f'{classifier}'}, inplace = True)
        else:
            train_df = pd.concat([train_df, group['Training time']], axis = 1)
            test_df = pd.concat([test_df, group['Training time']], axis = 1)
            train_df.rename(columns = {'Training time': f'{classifier}'}, inplace = True)
            test_df.rename(columns = {'Training time': f'{classifier}'}, inplace = True)

        train_df.to_csv(f'{time_tests_path}/{dr_technique}_training.csv')
        test_df.to_csv(f'{time_tests_path}/{dr_technique}_testing.csv')


# %%
# Creating a dataframe for the 3 DR techniques I conducted time tests (fitting and then reduction times)
dr_techniques = ['PCA', 'NMF', 'AE']
train_df, reduction_df = pd.DataFrame(), pd.DataFrame()
for dr_technique in dr_techniques:
    
    aux_df = pd.read_csv(f'{dr_time_tests_path}/{dr_technique}_DR_Times.csv')
    aux_df = aux_df[['N_components', 'Fit time', 'Pred test time']]
    aux_df.reset_index(drop = True, inplace = True)
    
    if train_df.empty:
        train_df = aux_df[['N_components', 'Fit time']]
        train_df.rename(columns = {'Fit time': dr_technique}, inplace = True)
    if test_df.empty:
        reduction_df = aux_df[['N_components', 'Pred test time']]
        reduction_df.rename(columns = {'Pred test time': dr_technique}, inplace = True)
    else:
        train_df = pd.concat([train_df, aux_df['Fit time']], axis = 1)
        reduction_df = pd.concat([reduction_df, aux_df['Pred test time']], axis = 1)
        train_df.rename(columns = {'Fit time': dr_technique}, inplace = True)
        reduction_df.rename(columns = {'Pred test time': dr_technique}, inplace = True)

train_df.to_csv(f'{dr_time_tests_path}/DR_training_times.csv')
reduction_df.to_csv(f'{dr_time_tests_path}/DR_reduction_times.csv')


# %%

# Generating a table that compares the performance and time increases for multiple methods
dr_techniques = Configs['models']['available_models']

for dr_technique in dr_techniques:

    # Load the times and MCC dataframes
    mcc_df = pd.read_csv(f'{default_tests_path}/{dr_technique}_MCC.csv', index_col = 'Unnamed: 0')
    times_df = pd.read_csv(f'{time_tests_path}/{dr_technique}_training.csv', index_col = 'Unnamed: 0')
    times_df.rename(columns = {'LogisticRegression': 'LogReg', 'KNeighborsClassifier': 'kNN', 'MLPClassifier': 'MLP', 'RandomForestClassifier': 'RF'}, inplace = True)
    
    df = pd.DataFrame()
    # Iterating and generating a new table for each classifier essentially
    for classifier in short_classifiers_list:
        
        # Getting the MCC and time values for the classifier
        mcc_values = mcc_df[classifier]
        time_values = times_df[classifier]

        # Getting the maximum and minimum values for the MCC and time
        max_mcc, min_mcc = float(mcc_values.tail(1)), float(mcc_values.head(1))
        max_time, min_time = float(time_values.tail(1)), float(time_values.head(1))

        # Getting the total performance and time increases
        total_performance_increase = ((max_mcc - min_mcc) / min_mcc) * 100
        total_time_increase = ((max_time - min_time) / min_time) * 100

        # Getting the performance and time increases per component
        performance_increase = total_performance_increase/(mcc_values.shape[0] - 1)
        time_increase = total_time_increase/(time_values.shape[0] - 1)

        # Creating the auxiliary series
        aux_series = pd.Series([classifier, performance_increase, time_increase, total_performance_increase, total_time_increase], index = ['Classifier', 'Performance Increase (p/ comp.)', 'Time increase (p/ comp.)', 'Total performance increase', 'Total time increase'])
        df = pd.concat([df, aux_series.to_frame().T], axis = 0)
        
    df.to_csv(f'{time_tests_path}/{dr_technique}_Percentages.csv')



# # %%
# # creating another dataframe, this time comprised of multiple training and prediction times for several classifiers
# aux_df = pd.DataFrame()
# dims_1 = [2, 5, 10, 20, 30, 40]
# dims_2 = [2, 4, 8, 16, 32]

# for dr_technique in dr_techniques:

#     # Loading the times dataframe
#     times_df = pd.read_csv(f'{time_tests_path}/{dr_technique}_times.csv')
#     grouped = times_df.groupby('Classifier')


#     for classifier, group in grouped:

#         group.reset_index(drop = True, inplace = True)

#         if dr_technique == 'AE':
#             dims = dims_2
#         else:
#             dims = dims_1

#         for dim in dims:

#             a = round(float(group.loc[times_df['N_components'] == dim]['Training time']), 3)
#             b = round(float(group.loc[times_df['N_components'] == dim]['Training std']), 3)

#             c = round(float(group.loc[times_df['N_components'] == dim]['Testing time']), 3)
#             d = round(float(group.loc[times_df['N_components'] == dim]['Testing std']), 3)

#             aux_series = pd.Series([dr_technique, classifier, dim, str(a) + '+' + str(b), str(c) + '+' + str(d)], index = ['DR', 'Classifier', 'N_components', 'Training time', 'Testing time'])
#             aux_df = pd.concat([aux_df, aux_series.to_frame().T], axis = 0)

# aux_df.to_excel('aaa.xlsx')

# # %%
# # Same as previous, applied to the default testing and prediction times
# times_df = pd.read_csv('./Results/Times/Default_Times.csv')

# aux_df = pd.DataFrame()
# for row in times_df.iterrows():


#     row = row[1]
#     classifier = row['Classifier']
#     a = round(float(row['Training time']), 3)
#     b = round(float(row['Training std']), 3)

#     c = round(float(row['Testing time']), 3)
#     d = round(float(row['Testing std']), 3)


#     aux_series = pd.Series([classifier, str(a) + '+' + str(b), str(c) + '+' + str(d)], index = ['Classifier', 'Training time', 'Testing time'])
#     aux_df = pd.concat([aux_df, aux_series.to_frame().T], axis = 0)

#     aux_df.to_excel('bbb.xlsx')
# # %%
