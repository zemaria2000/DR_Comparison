# DR_Comparison

Repository containing the code with a comparison between a series of Dimensionality Reduction techniques. 

It is the repository connected to a paper developed in the scope of the IoTBDS conference (<b>paper is unavailable at the moment</b>)


## Use case description

This repository was used to test multiple types of DR techniques. These tests were conducted on a real (private) industrial dataset at <a href="https://www.bosch.pt/a-nossa-empresa/bosch-em-portugal/aveiro/" target="_blank">Bosch Termotecnologia Aveiro</a>. At the end of one of their boilers' production line, tests are conducted in order to assure no leaks are found in the boilers' piping systems. The tests involve gathering data variables such as gas flows, temperatures, and pressures (around 50). If the values fall between adequate thresholds, the tests are successful, otherwise, the test fails. When that does happen, manual examination is required in order to identify the leak's location, inadvertently leading to production bottlenecks. 

Within this scenario, dimensionality reduction techniques, combined with Machine Learning, can have a significant role in the optimisation of the testing procedure, identifying the main features and variables that can be input to ML classifiers, for them to promptly analyse if a test is successful or not.

A ML use case was created, where for 2 different boilers, a 4 class multi-labelling classification problem was created, with the labels defining the success of failure of each equipment's test. Then, different DR techniques were implemented in order to assess the impacts of reducing the amount of variables within the classifier's classification and time performance.


## Different repository possibilities

The main work, thoroughly described in the repository's paper (<b>not yet available</b>), involved the development of 3 different tests:

1.  Assessing the influence of different DR approaches with varying numbers of reduced features on classifier performance;
2.  Evaluating classifier fitting and prediction times for different dataset dimensions;
3.  Assessing fitting and dataset reduction times for some DR techniques.

To do so, this repository contains some essential python scripts and folders:

*  [Data_Pre_Processing.py](https://github.com/zemaria2000/DR_Comparison/blob/main/Data_Pre_Processing.py) - although the dataset is not made available, the pre-processing file is shared, giving a perception of what was done to prepare the datasets for processing;
*  [Results.py](https://github.com/zemaria2000/DR_Comparison/blob/main/Results.py) - file that is used to conduct the different tests
*  [Classes](https://github.com/zemaria2000/DR_Comparison/tree/main/Classes) - folder containing 2 classes, the most important being the [Testing.py](https://github.com/zemaria2000/DR_Comparison/tree/main/Classes/Testing.py), which helped in gathering the results for the paper

Not reflected in the official repository paper, but also developed under this project were the hyperparameter optimisation scripts. Those were the following:

*  [SearchParameters.py](https://github.com/zemaria2000/DR_Comparison/blob/main/SearchParameters.py) - Used to define the search space for the hyperparameter optimisation library that was used - <a href="https://scikit-optimize.github.io/stable/" target="_blank">scikit-optimize</a>
*  [Train.py](https://github.com/zemaria2000/DR_Comparison/blob/main/Train.py) - script that is used to build the optimised models, i.e., to conduct the DR techniques's optimisation process
*  [Optimise.py](https://github.com/zemaria2000/DR_Comparison/tree/main/Classes/Optimise.py) - a class, within the classes folder, which helps with the hyperparameter optimisation process

There are also other scripts and folders, but those were essentially related with the paper's development (organising data for some tables, plotting some results, etc)

## How to implement it

If you just want to test the default methods, do the following sequence:

1.  Run the `requirements.txt` file
    ```sh
    pip install -r requirements.txt
    ```

2.  Conduct the data pre-processing using the `Data_Pre_Processing.py` file

3.  Then run the tests through the `Results.py` file


If you wish to conduct hyperparameter optimisation, you should additionally (and before the previous step 3):

1.  Run the `SearchParameters.py` file, which will generate the list of hyperparamaters, for each Non-DL technique, to be optimised

2.  Run the `train.py` file, but with caution. If there is a particuiar DR model that you'd wish to modify, please run the command:
    ```sh
    python train.py <model>
    ```

3.  Then run the tests



### Full essential file/folder description

* Classes folder - folder containing 2 classes that were created for this particular testing procedure:
    * [Testing.py](https://github.com/zemaria2000/DR_Comparison/tree/main/Classes/Testing.py) - class that is used to ease the testing process of the models. It has methods that fit the DR techniques, reduce the datasets, and even store the results;
    * [Optimise.py](https://github.com/zemaria2000/DR_Comparison/tree/main/Classes/Optimise.py) - contains a series of important methods that help with the hyperparameter optimisation process of the DR techniques, such as the optimisation functions for the DL and Non-DL methods, or the storing of the optimised models;

* [Data_Pre_Processing.py](https://github.com/zemaria2000/DR_Comparison/tree/main/Data_Pre_Processing.py) - file that is used to pre-process the private dataset used. It is specific for the dataset. Yet, it gives an idea of the main pre-processing steps conducted, from organising the original dataset, label encoding, to balancing the number of samples

* [SearchParameters.py](https://github.com/zemaria2000/DR_Comparison/tree/main/Classes/SearchParameters.py) - file that creates the search space for the 6 Non-DL DR methods, in case of optimisation. Each method has a different search space, and it's up to the user to define the search space he wishes to

* [train.py](https://github.com/zemaria2000/DR_Comparison/tree/main/train.py) - file that conducts the hyperparameter optimisation process. It makes use of the `Optimise.py` class that was created, and it can optimise all methods in a row or the method that the user specifies as a command argument

* [Results.py](https://github.com/zemaria2000/DR_Comparison/tree/main/Results.py) - this was the file that allowed to get the results for the tests that were conducted and are present in the repository's related paper. It has 4 different tests, and makes use of the `Testing.py` class. It can analyse, for each number of reduced components for each technique, the classifier's performance metrics; it allows for testing the optimised DR techniques as well; and then it allows for testing the execution times of both the classifiers and the DR techniques

* [ResultsProcessing.py](https://github.com/zemaria2000/DR_Comparison/tree/main/ResultsProcessing.py) - it just was created to better organise some information regarding the tests. Nothing else

* [Plots.py](https://github.com/zemaria2000/DR_Comparison/tree/main/Plots.py) - another "auxiliary" file, used to plot some graphs for the paper

* [settings.yaml](https://github.com/zemaria2000/DR_Comparison/tree/main/settings.yaml) - file containing a series of common values, strings and arrays used throughout the multiple files of the repository