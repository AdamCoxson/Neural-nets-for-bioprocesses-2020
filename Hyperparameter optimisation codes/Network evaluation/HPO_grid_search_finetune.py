# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 19:47:11 2020
Author: Adam Coxson, MPhys Undergrad, The University of Manchester
Project: Evaluating hyperparameter optimisation techinques for Bioprocesses
"Finding the Optimal Optimiser"
Module: HPO_grid_search_finetune
Dependancies: neural_networks, data_preprocessing,FNN_test_module,
              RNN_test_module, train_v2

This script performs grid search upon a defined list of hyperparameters for 
either an FNN or an RNN. The data s then written to a csv file. 
Note that parallel processing is used to evaluate the same set of 
hyperparameters several times (n_samples), this ensures that the outputted 
accuracy reading from the neural network is representative of the parameters.
The mean and standard deviations of these n_samples are outputted and saved.

Hyperparmeters
No. hidden layers - Fixed at 2 for the FNN, 1 for the RNN.
No. neurons per layer
Activation function - Fixed at sigmoid for the FNN, tanh for the RNN
Learning rate - the magnitude to update weights by during back propagation
Batch size - The number of datapoints to process before weights update
No. of EPOCHs - The number of iterations through the whole dataset 
EPOCHs are the number of network iterations required to train the network. 

"""

import numpy as np                             # Packages -----
import copy
import csv
from cpuinfo import get_cpu_info
from datetime import datetime
from joblib import Parallel, delayed
#import psutil # For joblib worker timeout/memory leak

from neural_networks import fnn_net # Modules 
from train_v2 import train
from FNN_test_module import FNN_test_mse_only
from RNN_test_module import RNN_test_mse_only
from data_preprocessing import data_preprocess

def parallel_network_evaluation(EPOCHS, lr, bs, network_type,net):
    """
    ARGS:    (Epoch number, learning rate, batch size, network tpye (either fnn or rnn), network to be evaluated).
    OUTPUTS: (Avg testing MSE value and training MSE value, from 4  testing data sets).
    
    This function takes in a set of hyperparameters and evaluates them for a neural network. This is built to
    be used with joblibs 'embarrasingly parallel' parallel procedure. This calls the training and testing
    functions for either the fnn or rnn.
    """
    training_mse = train(net, training_inputs, training_labels, EPOCHS, lr, bs, network_type)
    if network_type == 'fnn':        
        mse_val = FNN_test_mse_only(test_inputs, test_labels, net)
    elif network_type == 'rnn':
        mse_val = RNN_test_mse_only(test_inputs, test_labels, net)
    else:
        print("Invalid neural network type, choose either fnn or rnn")
    return (mse_val, training_mse)

def FNN_neural_net(cfg, cores, n_samples):
    """
    ARGS: (A single configuration of hyperparameters (H1,H2,EPOCHS,BS,LR), number of cores to be used in parallel processing (global), number of 
           samples to average over (global))
    OUTPUTS: (Averaged MSE error + half std dev of n_samples for the given hyperparameters).
    
    This function takes in a set of hyperparameters and evaluates them for the FNN architecture developed in the program. 
    It repeats the network training and testing n_samples times over and averages this data, utilising parallel
    processing and the parallel_network_evaluation() function. This is used to further evaluate optimal hyperparameters
    identifed from previous analyse. The purpose of this is to identify any outliers wrongly believed to be optimal as 
    many hyperparameter sets have a very high variance after training and testing. The parallel processing used is from 
    the joblib package which automatically deals with multiple processes and the data reorganisation after processing.
    """
    start_time = datetime.now()
    h1     = int(cfg[0])
    h2     = int(cfg[1])
    EPOCHS = int(cfg[2])
    bs     = int(cfg[3])
    lr     = float(cfg[4])
    mse_vals = np.zeros(n_samples)
    training_mse_vals = [0]*n_samples
    results = [0]*n_samples
    networks = [0]*n_samples
    
    for i in range(0,n_samples):
        net = fnn_net(h1, h2)
        init_state = copy.deepcopy(net.state_dict())
        net.load_state_dict(init_state)
        networks[i] = net
    # Joblib parallelisation across the number of samples
    results[:] = Parallel(n_jobs=cores)(delayed(parallel_network_evaluation)(EPOCHS, lr, bs, 'fnn', net) for net in networks)
    for i in range (0,n_samples):
        mse_vals[i] = results[i][0]
        training_mse_vals[i] = np.array(float(results[i][1]))

    train_mse = np.mean(training_mse_vals)
    train_std = np.std(training_mse_vals)
    avg_mse = np.mean(mse_vals)
    std_mse = np.std(mse_vals)
    end_time = datetime.now()

    print('\nParameters HN: (%d, %d), Epochs: %d, Batch Size: %d, Learn rate: %.6f'%(cfg[0],cfg[1],cfg[2],cfg[3],cfg[4]))
    #print("MSE values:\n",(mse_vals))
    print('Mean Squared Error: (%.4f \u00b1 %.4f), Mean Error: (%.4f \u00b1 %.6f)'%(avg_mse,std_mse,
                                                                    np.mean(np.sqrt(mse_vals)),np.std(np.sqrt(mse_vals))))
    print('Training Mean Squared Error: (%.4f \u00b1 %.4f),)'%(train_mse,train_std))
    print("Training execution time: ", (end_time-start_time), "(hrs:mins:secs)")
    return (avg_mse, std_mse, train_mse, train_std, mse_vals, (end_time-start_time))


def print_results(hyperparam_list, results, network_type):
    """
    ARGS: (list of hyperparameters, results corresponiding to the hyperparameters: avg_mse, std_mse, train_mse,
           train_std, mse_vals, mse_vals and runtime, network type- FNN or RNN).
    OUTPUTS: (no function return, outputs text to console for Copy pasta).
    
    This function displays the results from evaluation of optimal hyperparameters obtained from previous analysis.
    The outputs are the hyperparameters with the corresponding mse vals for each of the n_samples. It also gives
    the mean and standard deviation for the absolute mean error and the mean squared error.
    """
    for i in range(0,len(hyperparam_list)):
        cfg = hyperparam_list[i]
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
        print('Training Mean Squared Error: (%.4f \u00b1 %.4f),)'%(train_MSE,train_std))
        print("Training execution time: ", time, "(hrs:mins:secs)")

    
def write_to_csv(filename, results, x_iters, hyperparameter_names, n_samples, time, cores):
    
    for i in range(0,len(results)):
        x_iters[i].insert(0,results[i][3])
        x_iters[i].insert(0,results[i][2])
        x_iters[i].insert(0,results[i][1])
        x_iters[i].insert(0,results[i][0])
    
    writer = csv.writer(open(filename,'w'),lineterminator ='\n')
    writer.writerow(["Time:", time, "Samples",n_samples, "Size:", len(results),"Cores:",cores,"CPU:", get_cpu_info()['brand_raw']])
    writer.writerow(hyperparameter_names)
    writer.writerows(x_iters)
    print("Written to",filename,"successfully.")


# Grid Search Training Loop

# HN = [(2,2),(4,2),(6,2),(8,2),(10,2), (2,4), (4,4), (6,4), (8,4), (10,4)] # size 10
# EPOCHS = [ 10,  50,  100, 150, 200, 250, 300, 350, 400]  #size 9
# batch_size = [ 1, 50, 100, 150, 200, 250, 300] # size 7
# LR = [0.0001, 0.0002, 0.0004, 0.001, 0.002, 0.004, 0.01, 0.02, 0.04, 0.1] # size 10

# HN = [(2,2),(4,2),(6,2),(8,2),(10,2), (2,4), (4,4), (6,4), (8,4), (10,4)] # size 10
# EPOCHS = [ 10,  25, 50,  100, 150, 200, 250, 300, 350, 400] #size 10
# batch_size = [ 1, 10, 25, 50, 80, 100, 150, 200, 250, 300] # size 10
# LR = [0.0001, 0.0002, 0.0004, 0.001, 0.002, 0.004, 0.01, 0.02, 0.04, 0.1] # size 10

#EPOCHS = np.linspace(0,500,51)
#batch_size = np.linspace(0,500, 51)
# batch_size = np.linspace(1,300, 14)
#LR = np.logspace(-4, -1, num = 51)
#LR = np.round(LR[:],6)
#EPOCHS[0] = batch_size[0] = 1

#H1 = [2, 4, 6, 8, 10, 12]
#H2 = [2, 4, 6, 8, 10, 12]

# EPOCHS = np.linspace(10,300,11)
# LR = np.round(np.logspace(-4, -1, num = 19),6)

#EPOCHS = np.linspace(20,1000,99)

#EPOCHS = [10,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800]
EPOCHS = [510]
LR = [0.00015]
#LR = np.round(np.logspace(-4, -3, num = 30),6)
#LR = [0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
HN = [(8,4)]
#batch_size = [200]
batch_size = np.linspace(50,800,16)
#batch_size[0] = 1

hyperparam_list = []
for h in HN:
    for epochs in EPOCHS:
        for bs in batch_size:
            for lr in LR:
                hyperparam_list.append([h[0],h[1],epochs,bs,lr])
avg_mse_data = [0]*len(hyperparam_list)
mse_err = [0]*len(hyperparam_list)
train_mse = [0]*len(hyperparam_list)
train_err = [0]*len(hyperparam_list)
results = [0]*len(hyperparam_list)

print("\nscript started\n")
                
cores = 4
n_samples = 16
init_time = datetime.now()
training_data, testing_data, raw_testing_data, scaler_test = data_preprocess('fnn') 
training_inputs = training_data[:, 0:4]
training_labels = training_data[:, 4:]
test_inputs = testing_data[:, 0:4]
test_labels = testing_data[:, 4:]
for j in range(0,len(hyperparam_list)):
    results[j] = FNN_neural_net(hyperparam_list[j], cores, n_samples)
    avg_mse_data[j] = results[j][0]
    mse_err[j]      = results[j][1]
    train_mse[j]    = results[j][2]
    train_err[j]    = results[j][3]
    
fin_time = datetime.now()
#print_results(hyperparam_list, results, 'fnn') # Func call, prints evaluation outputs such as the means and std devs
print("\nGrid search algorithm execution time: ", (fin_time-init_time), "(hrs:mins:secs)")
print("No. of parameter configurations: ", len(hyperparam_list),"\n" )
        
# OUTPUTS # ------------------------------------
# sorted_indexes = np.argsort(avg_mse_data) # Returns a list of original indexes if the array were to be sorted
# for i in range (0,8):
#     min_index = sorted_indexes[i]
#     cfg = hyperparam_list[min_index]
#     print('%d) Parameters HN: (%d, %d), Epochs: %d, Batch Size: %d, Learn rate: %.4f'%(i+1,cfg[0],cfg[1],cfg[2],cfg[3],cfg[4]))
#     print("MSE: ", np.round(avg_mse_data[min_index],5), "\n")
#filename = 'Grid_BS_HN_8_4_epochs510_lr0p00015.csv'
hyperparam_names = ["MSE","std","Train MSE","Train std","H1","H2","Epochs", "Batch Size", "Learning rate"]
write_to_csv(filename, results, hyperparam_list, hyperparam_names, n_samples, fin_time-init_time, cores)


