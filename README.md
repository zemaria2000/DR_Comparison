# DR_Comparison

Repository containing the code with a comparison between a series of Dimensionality Rediction techniques. 

It follows the conceptual work I developed with my review article on dimensionality reduction techniques applied in industry.


## What is its purpose

This repository was used to test multiple types of DR techniques. These tests were conducted on a real (private) industrial dataset. At an industrial production line, tests were conducted to assess the pressure drop along different boilers' water circuits. However, those tests involved gathering up to 50/60 variables at the same time. At the end of the tests, each boiler would be given a result (1 or 2) depending if it passed or failed, respectively, the test.

Therefore, a ML use case was created. Taking on those initial variables, DR techniques were applied to first reduce the dimensions and complexity of the tests. Then, a classifier was applied to the reduced data in order to assess the ability to classify the tests on the reduced dataset.

This repository conducts DR tests for multiple techniques, hoping to assess the different methods' performances


## How does it work

According to the dataset we wish to implement, there is a certain order for the scripts to be ran:

1.  The first script should be the `Data_Pre_Processing.py`, which will prepare the data to be processed
2.  Then, there is also the `SearchParameters.py`, which generates and saves different hyperparameters for different models to be optimised
3.  After that, the `train.py` script should be ran, but with caution:

``` python train.py <model>``` 

4.  Finally, the test can be conducted