README for Neural Network Code - FNN and RNN

Author: Adam Coxson, MPhys Undergrad, The University of Manchester
Email: adamcoxson1@gmail.com
Project: Evaluating hyperparameter optimisation techinques for an FNN and RNN, Aug 2020
Reports: 
Main report comparing FNN and RNN performance (Polished): https://www.overleaf.com/read/xctrqdqybckj
Secondary report comparing hyperparameter optimisation method (very unpolished): https://www.overleaf.com/read/hdmdcfkqyrpx
Google drive for entire project: https://drive.google.com/drive/folders/14E8SYFzoncRtdEVwSguZaxXrR-hf7ufy?usp=sharing

This code was for my SEI internship project into hyperparameter optimisation techniques
and neural networks for bio-processes. My supervisor was Dr Dongda Zhang from the 
School of Chemical Engineering and Analytical Sciences, Uni of Manchester.
Email: dongda.zhang@manchester.ac.uk

This folder contains 4 main scripts (denoated by the 'ann_' prefix) and 5 other .py file modules.
Also in here are experimental data csv files for an algae growth bioprocess.
The folder "Data plotting functions" has some scripts that show how I plotted the testing
and training MSE in 1D plots as well as my heatmap plot codes.

A note on the experimental data - the original csv file has data entries missing and v2 has 
these entries filled in by copying the previous data. This doesn't matter as the final
values at the 12th hour are deleted anyway so aren't used for network training or testing.
This was simply done to make my life easier in the data preprocessing, rather than have to
check each experimental trial to see whether it was 10, 11 or 12 entries long.

A note on Joblib parallelisation.
When trying to use more than one core, I'd often get this error:
"BrokenProcessPool: A task has failed to un-serialize. 
Please ensure that the arguments of the function are all picklable."
Since I used Spyder 4.2, my way around it was to open a special Pylab console/kernal. 
I've also seen suggestions that running the script directly from the command line or
by using a linux OS, you don't have this problem, its specific to windows. The joblib
authors are aware of this issue.
https://joblib.readthedocs.io/en/latest/auto_examples/serialization_and_wrappers.html
https://github.com/scikit-learn/scikit-learn/issues/12413

ann_bayesian_optimisation.py
This script performs Bayesian optimisation for the FNN and RNN architectures from my project.
This will allow you to identify an optimal set of hyperparameters. In this script I've also 
provided links and examples of how to apply random forest and random search using the
skopt.optimize package.

ann_grid_search.py
This is used for hyperparameter optimisation by performing a grid search algorithm.
This is quite useful for obtaining data for 1D plots or 2D heat map plots

ann_hyperparameter_evaluation.py
This is just a simple script that trains and tests the neural network on pre-defined
configurations of hyperparameters. It then plots the prediction plots for further assessment
For example, find a good optimal set from the Bayesian script, then evaluate with
n_samples = 100 in this script to get a very accurate mean and standard deviation.

ann_random_search.py
Near identical format as the grid search scipt but randomly assigns the hyperparameter
configuration fields.

data_preprocessing.py
A function module which formats and normalises the current experimental data to be compatible
with the ANNs. Also replicates the data with random seeding to artifically increase the amount
of training data available. If you want to use the current neural network architectures with
different experimental data, reformat this module to be more general or create a 2nd version.

Below are all the modules which should work out the box for different data provided the current 
network architectures are still used.
network_train_and_test_module.py - brings the neural networks and train test functions together
neural_networks.py - RNN and FNN class codes using the PyTorch package functionality
testing_functions.py
training_function.py

