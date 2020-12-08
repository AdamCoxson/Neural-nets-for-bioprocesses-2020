# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 21:19:29 2020
Author: Adam Coxson, MPhys Undergrad, The University of Manchester
Project: Evaluating hyperparameter optimisation techinques for an FNN and RNN
Module: HPO_Bayesian_finetune  (Gaussian Process Bayesian Optimisation)
Dependancies: neural_networks, data_preprocessing, FNN_test_module,
              RNN_test_module, train_v2

This script performs Gaussian Process Bayesian Optimisation for either an FNN
or RNN neural network architecture. The results are written to an excel file,
execution times (for 250 evaluations) can last from 1 hour to 12 hours.
Note that parallel processing is used to evaluate the same set of 
hyperparameters several times (n_samples), this ensures that the outputted 
accuracy reading from the neural network is representative of the parameters.
The mean of these is taken and half the standard deviation is also added on,
this helps to bias the output so gp_minimize selects parameters with small 
deviations from their mean.

Resources:
https://machinelearningmastery.com/what-is-bayesian-optimization/

Note, to ensure use_named_args works, I have had to place the line @use_named_args(bounds)
in front of the definitions for F/RNN_neural_network(), as well as the definition for
bounds.

"""

import copy
import numpy as np
import csv
import warnings
from cpuinfo import get_cpu_info
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from datetime import datetime
from joblib import Parallel, delayed

from neural_networks import fnn_net, rnn_net
from train_v2 import train
from FNN_test_module import FNN_test_mse_only
from RNN_test_module import RNN_test_mse_only
from data_preprocessing import data_preprocess


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# This filters the following warnings (tbh i don't really know what it means):
# C:\ProgramData\Anaconda3\lib\site-packages\torch\storage.py:34: FutureWarning: pickle support for Storage will be removed in 1.5. Use `torch.save` instead
#  warnings.warn("pickle support for Storage will be removed in 1.5. Use `torch.save` instead", FutureWarning)
# C:\ProgramData\Anaconda3\lib\site-packages\joblib\externals\loky\process_executor.py:691: UserWarning: A worker 
# stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak



def parallel_network_evaluation(EPOCHS, lr, bs, network_type, net):
    """
    ARGS:    (Epoch number, learning rate, batch size, network type (either fnn or rnn), network to be evaluated).
    OUTPUTS: (MSE value, 5 mse values from 5 data sets used as testing data, predictions for the 5 test sets).
    
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

def write_to_csv(filename, results, x_iters, hyperparameter_names,n_samples, time, cores):
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
        x_iters[i].insert(0,results[i][3])
        x_iters[i].insert(0,results[i][2])
        x_iters[i].insert(0,results[i][1])
        x_iters[i].insert(0,results[i][0])
    
    writer = csv.writer(open(filename,'w'),lineterminator ='\n')
    writer.writerow(["Time:", time, "Samples",n_samples, "Size:", len(results),"Cores:",cores,"CPU:", get_cpu_info()['brand_raw']])
    writer.writerow(hyperparameter_names)
    writer.writerows(x_iters)
    print("Written to",filename,"successfully.")

random_vals = 50
num_evaluations = random_vals + 200
# random_vals = 2
# num_evaluations = random_vals + 1
cores = 4
n_samples = 16
all_config_results = [0]*num_evaluations
hyperparam_idx = 0
network_type = 'rnn'
if network_type == 'fnn':
    print("\nScript Started for the FNN.\n")
    bounds = [Categorical([2,4,6,8,10], name = 'H1'),
              Categorical([2,4], name = 'H2'),
              Integer(200, 800,  name='EPOCHS'), 
              #Integer(1, 400,  name='BS'),
              Categorical([100,200,300,400], name = 'BS'),
              Real(0.0001,0.02,"log-uniform",  name='LR')]
    # bounds = [                               # A quick set for debug
#           Categorical([2,4], name = 'H1'),
#           Categorical([2,4], name = 'H2'),
#           Integer(10, 50,  name='EPOCHS'), 
#           Categorical([200,250,300,350,400], name = 'BS'),
#           Real(0.0001,0.02,"log-uniform",  name='LR')
#           ]

    training_data, testing_data, dummy_var, dummy_var2 = data_preprocess('fnn') # Data formation
    training_inputs = training_data[:, 0:4]
    training_labels = training_data[:, 4:]
    test_inputs = testing_data[:, 0:4]
    test_labels = testing_data[:, 4:]
    
elif network_type == 'rnn':
    print("\nScript Started for the RNN.\n")
    bounds = [Integer(1, 30,  name='HN'),
              Integer(1, 200,  name='EPOCHS'), 
              Integer(1, 600,  name='BS'),
              #Categorical([20, 50, 80, 100,150,200,300,400,500], name = 'BS'),
              Real(0.0001,0.02,"log-uniform",  name='LR')]
        
    training_data, testing_data, dummy_var, dummy_var2 = data_preprocess('rnn')
    training_inputs = training_data[:,0:4]
    training_labels = training_data[:,4:]
    test_inputs = testing_data[:,0:4]
    test_labels = testing_data[:,4:]
    training_inputs = np.split(training_inputs, 861)
    training_labels = np.split(training_labels, 861)
    test_inputs = np.split(test_inputs, 4)
    test_labels = np.split(test_labels, 4)
else:
    print("Invalid neural network type, choose either fnn or rnn")
          

@use_named_args(bounds) # Have to define it here for FNN_neural_net
def FNN_neural_net(H1, H2, EPOCHS, BS, LR):
    """
    ARGS: A single configuration of hyperparameters (H1,H2,EPOCHS,BS,LR),
          number of cores to be used in parallel processing (Passed as a global var),
          number of samples to average over (passed as a global var )
    OUTPUTS: Averaged MSE error + half std dev of n_samples for the given hyperparameters.
    
    This function takes in a set of hyperparameters and evaluates them for the FNN architecture developed in the program. 
    It repeats the network training and testing n_samples times over and averages this data, utiliting parallel
    processing and the parallel_network_evaluation() function. This is used to further evaluate optimal hyperparameters
    identifed from previous analyse. The purpose of this is to identify any outliers wrongly believed to be optimal as 
    many hyperparameter sets have a very high variance after training and testing. The parallel processing used is from 
    the joblib package which automatically deals with multiple processes and the data reorganisation after processing.
    """
    global hyperparam_idx
    global cores
    global n_samples
    global all_config_results

    mse_vals = np.zeros(n_samples)
    training_mse_vals = [0]*n_samples
    results = [0]*n_samples
    networks = [0]*n_samples
    
    for i in range(0,n_samples): # Building a list of networks to avoid retraining of a single, referenced network
        net = fnn_net(H1, H2)
        init_state = copy.deepcopy(net.state_dict())
        net.load_state_dict(init_state)
        networks[i] = net
    # Joblib parallelisation across the number of samples
    results[:] = Parallel(n_jobs=cores)(delayed(parallel_network_evaluation)(EPOCHS, LR, BS, 'fnn', net) for net in networks)

    for i in range (0,n_samples):
        mse_vals[i] = results[i][0]
        training_mse_vals[i] = np.array(float(results[i][1]))
    train_mse = np.mean(training_mse_vals) 
    train_std = np.std(training_mse_vals)
    avg_mse = np.mean(mse_vals)
    std_mse = np.std(mse_vals)
    all_config_results[hyperparam_idx] = [avg_mse, std_mse, train_mse, train_std] # Getting all the averaged data
    hyperparam_idx = hyperparam_idx + 1
    print('HN (%d, %d), Epochs %d, BS %d, LR %.6f, GP_output %.4f, MSE (%.4f \u00b1 %.4f), Train MSE (%.4f \u00b1 %.4f)'% (H1, H2,
                                                            EPOCHS, BS, LR,avg_mse+(std_mse/2), avg_mse, std_mse, train_mse, train_std))
    # Return the average mse plus some of its error. Biases gp_minimize to use smaller stds
    return avg_mse+(std_mse/2)

@use_named_args(bounds) # Have to define it here for RNN_neural_net
def RNN_neural_net(HN, EPOCHS, BS, LR):
    """
    ARGS: A single configuration of hyperparameters (HN,EPOCHS,BS,LR),
          number of cores to be used in parallel processing (Passed as a global var),
          number of samples to average over (passed as a global var )
    OUTPUTS: Averaged MSE error + half std dev of n_samples for the given hyperparameters.
    
    This function takes in a set of hyperparameters and evaluates them for the RNN architecture developed in the program. 
    It repeats the network training and testing n_samples times over and averages this data, utiliting parallel
    processing and the parallel_network_evaluation() function. This is used to further evaluate optimal hyperparameters
    identifed from previous analyse. The purpose of this is to identify any outliers wrongly believed to be optimal as 
    many hyperparameter sets have a very high variance after training and testing. The parallel processing used is from 
    the joblib package which automatically deals with multiple processes and the data reorganisation after processing.
    """
    global hyperparam_idx
    global cores
    global n_samples
    global all_config_results
    mse_vals = np.zeros(n_samples)
    training_mse_vals = [0]*n_samples
    results = [0]*n_samples
    networks = [0]*n_samples
    
    for i in range(0,n_samples): # Building a list of networks to avoid retraining of a single, referenced network
        HL = 1 # one layer RNN
        net = rnn_net(1, 4, 10, HN, HL)
        init_state = copy.deepcopy(net.state_dict())
        net.load_state_dict(init_state)
        networks[i] = net
    # Joblib parallelisation across the number of samples
    results[:] = Parallel(n_jobs=cores)(delayed(parallel_network_evaluation)(EPOCHS, LR, BS, 'rnn', net) for net in networks) 
    
    for i in range (0,n_samples):
        mse_vals[i] = results[i][0]
        training_mse_vals[i] = np.array(float(results[i][1]))
    train_mse = np.mean(training_mse_vals) 
    train_std = np.std(training_mse_vals)
    avg_mse = np.mean(mse_vals)
    std_mse = np.std(mse_vals)
    all_config_results[hyperparam_idx] = [avg_mse, std_mse, train_mse, train_std] # Getting all the averaged data
    hyperparam_idx = hyperparam_idx + 1
    print('HN %d, Epochs %d, BS %d, LR %.5f, GP_output %.4f, MSE (%.4f \u00b1 %.4f), Train MSE (%.4f \u00b1 %.4f)'% (HN,
                                                            EPOCHS, BS, LR, avg_mse+(std_mse/2), avg_mse, std_mse, train_mse, train_std))
    # Return the average mse plus some of its error. Biases gp_minimize to use smaller stds
    return avg_mse+(std_mse/2)

if network_type == 'fnn':
    func = FNN_neural_net
if network_type =='rnn':
    func = RNN_neural_net
    
# GAUSSIAN PROCESS BAYESIAN OPTIMISATION TUNING PROCEDURE
init_time = datetime.now()
gp_output = gp_minimize(func,    # the function to minimize
                     dimensions=bounds,    # the bounds on each hyperparameter dimension 
                     acq_func="EI",        # the acquisition function (Expected improvement)
                     n_calls = num_evaluations,      # the number of evaluations of func
                     n_random_starts = random_vals,  # the number of random initialization points                       
                     random_state=1234)     # the random seed
fin_time = datetime.now()
print("\nBayesian GP Optimisation execution time: ", (fin_time-init_time), "(hrs:mins:secs)")

if network_type == 'fnn':
    print('Results: MSE: %.4f, HN: (%d, %d), Epochs: %d, Batch Size: %d, Learn rate: %.6f' % (gp_output.fun,
                              gp_output.x[0], gp_output.x[1],gp_output.x[2], gp_output.x[3],gp_output.x[4]))
    hyperparams = ["MSE","std","Train MSE","Train std","H1","H2","Epochs", "Batch Size", "Learning rate"]
    
if network_type =='rnn':
    print('Results: MSE: %.4f, HN: %d, Epochs %d, Batch Size %d, Learn rate %.6f' % (gp_output.fun,
                                    gp_output.x[0], gp_output.x[1], gp_output.x[2], gp_output.x[3]))
    hyperparams = ["MSE","std","Train MSE","Train std","HN","Epochs", "Batch Size", "Learning rate"]
    
filename = 'RNN_Bayesian_fine_A3'+'.csv'
write_to_csv(filename,all_config_results, gp_output.x_iters, hyperparams,n_samples,fin_time-init_time, cores)




    