# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 16:54:01 2020
Author: Adam Coxson, MPhys Undergrad, The University of Manchester
Project: Evaluating hyperparameter optimisation techinques for an FNN and RNN
Module: network_train_and_test_module 
Dependancies: training_function, testing_functions

This contains functions which are called in main scripts denoted by 'ann': ann_script_type.py.
These are required for training and testing of neural networks. train_and_test() is written to 
be compatible with joblib parallel processing which is implemented in the main scripts.

"""
# Packages 
import numpy as np    
import csv     
from cpuinfo import get_cpu_info          
# Modules         
from training_function import train
from testing_functions import FNN_test, RNN_test, FNN_test_mse_only, RNN_test_mse_only

def closest_to_average(vals, avg):
    """
    ARGS: 
        vals: a list of mse values from training and testing multiple times,
        avg: the average value of the list.
    OUTPUTS:
        sorted_idx[0]: The index of the value in the 'vals' list which is closest to 'avg').
    
    This function loops through 'vals' and returns the value which is closest to the average.
    This selects predicition data which is most representative of the hyperparameters.
    A similar version could be made which returns the smallest mse of the list.
    """
    diff_from_avg = np.sqrt((vals - avg)**2)
    sorted_idx = np.argsort(diff_from_avg)
    #print("\nDiff:",diff_from_avg, "Idx:",sorted_idx,"\n")
    return sorted_idx[0]

def train_and_test(EPOCHS, lr, bs, data_input, network_type, net):
    """
    ARGS: 
        EPOCHS: Epoch number,
        lr: learning rate,
        bs: batch size,
        data_input: training and testing data from pre-processing module,
        network_type: string for either 'fnn' or 'rnn'.
    OUTPUTS:  (outputted as a tuple, necessary for joblib parallel function)
        mse_val: mse value obtained from averaging the testing set data,
        training_mse: average mse value from passing through the training data
        testing_set_data: mse values for the individual testing sets,
        prediction_data: the networks prediction for each of the testing sets.

    This function takes in a set of hyperparameters, trains the network on them and 
    then tests the network on the training data. This works for either the fnn or
    the rnn and also returns the prediction data from the testing. This is built to
    be used with joblibs 'embarrasingly parallel' parallel procedure. 
    """
    training_inputs = data_input[0]
    training_labels = data_input[1]
    test_inputs = data_input[2]
    test_labels = data_input[3]
    training_mse = train(net, training_inputs, training_labels, EPOCHS, lr, bs, network_type)
    if network_type == 'fnn':        
        mse_val, testing_set_data, prediction_data = FNN_test(test_inputs, test_labels, net)
    elif network_type == 'rnn':
        mse_val, testing_set_data, prediction_data = RNN_test(test_inputs, test_labels, net)
    else:
        print("Invalid neural network type, choose either fnn or rnn")
    return (mse_val, training_mse, testing_set_data, prediction_data)
    
def train_and_test_mse_only(EPOCHS, lr, bs, data_input, network_type, net):
    """
    ARGS: 
        EPOCHS: Epoch number,
        lr: learning rate,
        bs: batch size,
        data_input: training and testing data from pre-processing module,
        network_type: string for either 'fnn' or 'rnn'.
    OUTPUTS:  (outputted as a tuple, necessary for joblib parallel function)
        mse_val: mse value obtained from averaging the testing set data,
        training_mse: average mse value from passing through the training data

    This function takes in a set of hyperparameters, trains the network on them and 
    then tests the network on the testing data. This works for either the fnn or
    the rnn. This is built to be used with joblibs 'embarrasingly parallel' parallel procedure. 
    """
    training_inputs = data_input[0]
    training_labels = data_input[1]
    test_inputs = data_input[2]
    test_labels = data_input[3]
    training_mse = train(net, training_inputs, training_labels, EPOCHS, lr, bs, network_type)
    if network_type == 'fnn':        
        mse_val = FNN_test_mse_only(test_inputs, test_labels, net)
    elif network_type == 'rnn':
        mse_val = RNN_test_mse_only(test_inputs, test_labels, net)
    else:
        print("Invalid neural network type, choose either fnn or rnn")
    return (mse_val, training_mse)
        
def write_to_csv(filename, results, x_iters, hyperparameter_names, n_samples, time, cores):
    """
    ARGS: filename: Name of the csv file to write the data to,
          results: a list of differnet MSE results for all hyperparameter sets,
          x_iters: the values for each hyperparameter passed in using the 'bounds' variable,
          hyperparameter_names: A list of all the hyperparameters used,
          n_samples: number of iterations of training and testing which were then averaged,
          time: total script execution time,
          cores: number of cores used to do multiple iterations of n_samples simultaneously,
    OUTPUTS: none, writes a csv file to cwd.
    
    This function takes in all the necessary details you would want recorded to use in later analysis
    """
    for i in range(0,len(results)):
        x_iters[i].insert(0,results[i][3]) # Avg MSE from n_samples
        x_iters[i].insert(0,results[i][2]) # MSE std dev
        x_iters[i].insert(0,results[i][1]) # Avg Training MSE fron n_samples
        x_iters[i].insert(0,results[i][0]) # Training MSE std dev
    
    writer = csv.writer(open(filename,'w'),lineterminator ='\n')
    writer.writerow(["Time:", time, "Samples",n_samples, "Size:", len(results),"Cores:",cores,"CPU:", get_cpu_info()['brand_raw']])
    writer.writerow(hyperparameter_names)
    writer.writerows(x_iters)
    print("Written to",filename,"successfully.")
     
def print_results(hyperparam_list, results, network_type):
    """
    ARGS: 
        hyperparam_list: a 2D list of different hyperparameters configurations,
        results: results corresponiding to the hyperparameters: avg_mse, std_mse, train_mse,
                 train_std, mse_vals, mse_vals and runtime,
        network_type: string for either fnn or rnn.
    OUTPUTS: 
        (no function return, outputs text to console for Copy pasta).
    
    This function displays the results from evaluation of optimal hyperparameters obtained 
    from previous analysis. It also gives the mean and standard deviation for 
    the absolute mean error and the mean squared error.   
    NOTE: This is currently redundant as the results are printed after each iteration anyway.
    """
    for i in range(0,len(hyperparam_list)):
        cfg = hyperparam_list[i] # a particular configuration of hyperparameters
        mean_MSE   = results[i][0]
        std_MSE    = results[i][1]
        train_MSE  = results[i][2]
        train_std  = results[i][3]
        mse_vals   = results[i][4]
        time       = results[i][5]
        mean_error = np.mean(np.sqrt(mse_vals))
        std_error  = np.std(np.sqrt(mse_vals))
                
        if network_type == 'fnn':
            print('\nParameters HN: (%d, %d), Epochs: %d, Batch Size: %d, Learn rate: %.6f'%(cfg[0],cfg[1],cfg[2],cfg[3],cfg[4]))
        if network_type == 'rnn':
            print('\nParameters HN: %d, Epochs: %d, Batch Size: %d, Learn rate: %.6f'%(cfg[0],cfg[1],cfg[2],cfg[3]))
        #print("MSE values:\n",(mse_vals))
        print('Mean Squared Error: (%.4f \u00b1 %.4f), Mean Error: (%.4f \u00b1 %.4f)'%(mean_MSE,std_MSE,mean_error,std_error))
        print('Training Mean Squared Error: (%.4f \u00b1 %.4f)'%(train_MSE,train_std))
        print("Training execution time: ", time, "(hrs:mins:secs)")
        