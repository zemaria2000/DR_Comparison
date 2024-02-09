# %%
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import json

# Loading the original dataframe
df = pd.read_csv('Data/OriginalDataset.csv', index_col = 'Unnamed: 0')
print(df.info())

# %%
# Converting all '-' values to NaNs
df.replace('-', np.nan, inplace = True)

# Converting all possible columns to floats (except for the equipment, test_results and new_results columns)
for col in df.columns:
    if col not in ['Eqwuipment', 'Test_Results', 'New_Results']:
        try:
            df[col] = df[col].astype(float)
        except:
            pass

# %%
# Replacing all NaN values from the columns by a value according to the columns' mean
for col in df.columns:
    if df[col].dtype == 'float64':
        df[col].fillna(df[col].mean(), inplace = True)


# %%
# Generating the X and y dataframes for a multilabel and binary classification problem
X = df.drop(columns = ['Equipment', 'Test_Results', 'New_Results'], axis = 1)
y = df['New_Results']
y_binary = df['Test_Results']
y_binary.replace(1, 0, inplace = True)
y_binary.replace(2, 1, inplace = True)
df = df.drop(columns = ['Equipment', 'Test_Results'], axis = 1)

# %%
# Using label encoder to convert the y labels into integer values
y_classes = LabelEncoder()
y = y_classes.fit_transform(y)
# Just saving the classes to store in a json file - https://bobbyhadz.com/blog/python-typeerror-object-of-type-int64-is-not-json-serializable
y_mapping = dict()
for elem in y_classes.classes_:
    y_mapping[elem] = int(y_classes.transform([elem]))
# y_mapping = dict(zip(y_classes.classes_, y_classes.transform(y_classes.classes_)))
with open('Data/y_classes.json', 'w') as file:
    file.write(json.dumps(y_mapping, indent = 4))

# Generating some scaled datasets and storing those
# NOTE: MinMaxScaler works best for autoencoders, while other methods such as PCA require centred data, such as the one generated by StandardScaler
standard_scaler_data = StandardScaler().fit_transform(X)
min_max_scaler_data = MinMaxScaler().fit_transform(X)

# Storing multiple pre-processed files to then be used by the models
np.savetxt('Data/X_StandardScaler.csv', standard_scaler_data, delimiter = ',')
np.savetxt('Data/X_MinMaxScaler.csv', min_max_scaler_data, delimiter = ',')
np.savetxt('Data/X_Original.csv', X, delimiter = ',')
np.savetxt('Data/y_LabelEncoder.csv', y, delimiter = ',')
np.savetxt('Data/y_Original', y, delimiter = ',')
np.savetxt('Data/y_Binary.csv', y_binary, delimiter = ',')
df.to_csv('Data/OrganisedOriginalDataset.csv')


# Creating also a binary classification problem
