""" Just a file that helped with the elaboration of the plots for the paper.
It is not relevant for the project itself """


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Importing the relevant data

# Graph 1 - ICA MCC for all components
ica_mcc = pd.read_csv('./Results/Default_Tests/ICA_MCC.csv', index_col = 'Unnamed: 0')
ica_mcc.drop(ica_mcc.tail(1).index, inplace = True)
# Graph 2 - RFE MCC for all components
rfe_mcc = pd.read_csv('./Results/Default_Tests/RFE_MCC.csv', index_col = 'Unnamed: 0')
rfe_mcc.drop(rfe_mcc.tail(1).index, inplace = True)
# Graph 3 - MLP training times evolution
mlp_training = pd.read_csv('Results/Times/ica_training.csv', index_col = 'Unnamed: 0')
mlp_training = mlp_training[['N_components', 'MLPClassifier']]

# %%

sns.set_palette('Set1')
sns.set_style('ticks')
sns.set_context(rc = {'font.size': 18,  'legend.fontsize': 18})
plt.figure(figsize=(9, 6))

# PLOT 1 - ICA MCC FOR ALL COMPONENTS
# sns.set_palette('Set1')
plt.figure(figsize=(9, 6))
sns.lineplot(x = ica_mcc['N_components'], y = ica_mcc['LogReg'], marker = 'o', label = 'LogReg', markersize = 8)
sns.lineplot(x = ica_mcc['N_components'], y = ica_mcc['kNN'], marker = '^', label = 'kNN', markersize = 10)
sns.lineplot(x = ica_mcc['N_components'], y = ica_mcc['MLP'], marker = 's', label = 'MLP', markersize = 8)
sns.lineplot(x = ica_mcc['N_components'], y = ica_mcc['RF'], marker = 'D', label = 'RF', markersize = 8)
plt.minorticks_on()
plt.tick_params(axis = 'x', which = 'minor', bottom = True, pad = 5, length = 2, width = 1)
plt.tick_params(axis = 'x', which = 'major', bottom = True, left = True, pad = 5, length = 5)
plt.xticks(np.arange(0,51,1), minor = True)
plt.xlabel('Number of Components')
plt.ylabel('MCC')
plt.grid(which = 'major', linestyle = '--', linewidth = 0.5)

plt.savefig('ICA_MCC.pdf', format = 'pdf', bbox_inches = 'tight')

# %%

# %%
# PLOT 2 - RFE MCC FOR ALL COMPONENTS
plt.figure(figsize=(9, 6))
sns.lineplot(x = rfe_mcc['N_components'], y = rfe_mcc['LogReg'], marker = 'o', label = 'LogReg', markersize = 8)
sns.lineplot(x = rfe_mcc['N_components'], y = rfe_mcc['kNN'], marker = '^', label = 'kNN', markersize = 10)
sns.lineplot(x = rfe_mcc['N_components'], y = rfe_mcc['MLP'], marker = 's', label = 'MLP', markersize = 8)
sns.lineplot(x = rfe_mcc['N_components'], y = rfe_mcc['RF'], marker = 'D', label = 'RF', markersize = 8)
plt.minorticks_on()
plt.tick_params(axis = 'x', which = 'minor', bottom = True, pad = 5, length = 2, width = 1)
plt.tick_params(axis = 'x', which = 'major', bottom = True, left = True, pad = 5, length = 5)
plt.xticks(np.arange(0,51,1), minor = True)
plt.xlabel('Number of Components')
plt.ylabel('MCC')
plt.grid(which = 'major', linestyle = '--', linewidth = 0.5)
plt.legend(loc = 'lower right')

plt.savefig('RFE_MCC.pdf', format = 'pdf', bbox_inches = 'tight')

# %%
# PLOT 3 - Training time for the MLP classifier
plt.figure(figsize=(9, 6))
sns.lineplot(x = mlp_training['N_components'], y = mlp_training['MLPClassifier']/1000, marker = 'o', markersize = 8)
plt.minorticks_on()
plt.tick_params(axis = 'x', which = 'minor', bottom = True, pad = 5, length = 2, width = 1)
plt.tick_params(axis = 'x', which = 'major', bottom = True, left = True, pad = 5, length = 5)
plt.xticks(np.arange(0,51,1), minor = True)
plt.xlabel('Number of Components')
plt.ylabel('Training time (s)')
plt.grid(which = 'major', linestyle = '--', linewidth = 0.5)

plt.savefig('MLP_Training.pdf', format = 'pdf', bbox_inches = 'tight')
# %%
