# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 15:49:38 2020
Author: Adam Coxson, MPhys Undergrad, The University of Manchester
Project: Evaluating hyperparameter optimisation techinques for an FNN and RNN
Module: ann_random_search
Dependancies: neural_networks, network_train_and_test_module, training_function, testing_functions, data_preprocessing

This script performs random search upon some defined ranges of hyperparameters for either an FNN or an RNN. 
Parallel processing is used to speed up the process. Two options for this are offered:
The 'samples' method means each configuration is evaluated in turn and the multiprocessing is applied at the 
training and testing loop, where the network is reset and evaluated n_sample times over.
The 'configurations' method uses multiprocessing to evaluate multiple configurations of hyperparameters at the
same time. This is useful for use in computer clusters thus having access to many more cores than n_samples.
Evaluating each set of hyperparameters several times over (n_samples) ensures that the outputted mean
error value is representative of the hyper-parameters, as some can have high variance.

Hyperparmeters
No. hidden layers - Fixed at 2 for the FNN, 1 for the RNN.
No. neurons per layer - HN, H1, H2. 
Activation function - Fixed at sigmoid for the FNN, tanh for the RNN
Learning rate - the magnitude to update weights by during back propagation
Batch size - The number of datapoints to process before weights update
No. of EPOCHs - A forward and backwards pass through the dataset 

"""
# Packages
import numpy as np                             
import copy
import random
from scipy.stats import loguniform
from datetime import datetime
from joblib import Parallel, delayed
# Modules
from neural_networks import fnn_net, rnn_net             
from network_train_and_test_module import train_and_test_mse_only, write_to_csv, print_results
from data_preprocessing import data_preprocess

def neural_network_evaluation(cfg, expt_data, cores, n_samples, network_type, parallel_type):
    """
    ARGS: cfg: A single configuration of hyperparameters (Neuron config, Epochs, batch size, learning rate),
          expt_data: formatted and normalised training and testing data  
          cores: number of cores to be used in parallel processing,
          n_samples: number of samples to average over for a given set of hyperparameters,
          network_type: either 'fnn' or 'rnn',
          parallel_type: either 'samples' or 'cnofigurations' dependent upon how to apply joblib.
          
    OUTPUTS: avg_mse: average of all the different, n_sample test mse values, 
             std_mse: spread of all the different, n_sample test mse values, 
             train_mse: average of all the different, n_sample training mse values,  
             train_std: spread of all the different, n_sample training mse values, 
             mse_vals: list of all the testing mse value (each an average of the test sets),
             Execution time.

    This function takes in a set of hyperparameters and evaluates them for the FNN or RNN architectures developed in the 
    program. It repeats the network training and testing n_samples times over and averages this data, utilising parallel
    processing via the train_and_test() function. Averaging over n_samples is necessary to get an accurate error estimate
    since different single iterations of train_and_test() can have high variance. It is recommended for n_samples to be
    10 to 20, parallel processing is present to help speed up the process.
    """
    start_time = datetime.now()
    if network_type == 'fnn': 
        H1     = int(cfg[0])
        H2     = int(cfg[1])
        EPOCHS = int(cfg[2])
        bs     = int(cfg[3])
        lr     = float(cfg[4])
    elif network_type == 'rnn':
        HL     = 1
        HN     = int(cfg[0])
        EPOCHS = int(cfg[1])
        bs     = int(cfg[2])
        lr     = float(cfg[3])
    else:
        print("Invalid neural network type, choose either fnn or rnn.")
        exit(1)
    mse_vals = np.zeros(n_samples)
    training_mse_vals = [0]*n_samples
    results = [0]*n_samples
    networks = [0]*n_samples
    
    for i in range(0,n_samples):
        if network_type =='fnn':
            net = fnn_net(H1, H2)
        else:
            net = rnn_net(1, 4, 10, HN, HL)     
        init_state = copy.deepcopy(net.state_dict())
        net.load_state_dict(init_state)
        networks[i] = net
        
    if parallel_type == 'samples':
        # Joblib parallelisation over the number of samples
        results[:] = Parallel(n_jobs=cores)(delayed(train_and_test_mse_only)(EPOCHS, lr, bs, expt_data, network_type, net) for net in networks)
    elif parallel_type == 'configurations':
        # Joblib parallelisation over different hyperparameter configs, occurs externally to this func
        for j in range(0,n_samples):
            results[j] = train_and_test_mse_only(EPOCHS, lr, bs, expt_data, network_type, networks[j])
    else:
        print("Invalid parallel processing method, choose either 'samples' or 'configurations'.")
        exit(1)
        
    for i in range (0,n_samples):
        mse_vals[i] = results[i][0]
        training_mse_vals[i] = np.array(float(results[i][1]))
    train_mse = np.mean(training_mse_vals)
    train_std = np.std(training_mse_vals)
    avg_mse   = np.mean(mse_vals)
    std_mse   = np.std(mse_vals)
    end_time  = datetime.now()

    if network_type == 'fnn':
        print('HN (%d, %d), Epochs %d, BS %d, LR %.6f, MSE (%.4f \u00b1 %.4f), Train MSE (%.4f \u00b1 %.4f)'% (H1, H2,
                                                EPOCHS, bs, lr, avg_mse, std_mse, train_mse, train_std))
    if network_type == 'rnn':
        print('HN %d, Epochs %d, BS %d, LR %.6f, MSE (%.4f \u00b1 %.4f), Train MSE (%.4f \u00b1 %.4f)'% (HN,
                                                EPOCHS, bs, lr, avg_mse, std_mse, train_mse, train_std))
    print("Training and testing execution time", (end_time-start_time), "(hrs:mins:secs)\n")
    return (avg_mse, std_mse, train_mse, train_std, mse_vals, (end_time-start_time))


# # # # # MAIN # # # # 
# Random Search 

cores = 4
n_samples = 10
combinations = 5 # number of hyperparameter configurations
network_type = 'fnn'
parallel_type = 'samples'
#parallel_type = 'configurations'


HN = [(2,2),(4,2),(6,2),(8,2),(10,2), (2,4), (4,4), (6,4), (8,4), (10,4)] # FNN
# HN = np.linspace(1,30,30) # RNN
hyperparam_list = []
for i in range(0,combinations):
    hn = random.choice(HN)
    epochs = np.random.randint(low=100, high=800)
    bs = np.random.randint(low=1, high=500)
    lr = round(loguniform.rvs(10**-4, 1*(10**-1)),5)
    if network_type == 'fnn':
        hyperparam_list.append([hn[0], hn[1], epochs, bs, lr])
    elif network_type == 'rnn':
        hyperparam_list.append([hn, epochs, bs, lr])
    else:
        print("Invalid network type, choose either 'fnn' or 'rnn'.")
        exit(1)

avg_mse_data = [0]*len(hyperparam_list)
mse_err = [0]*len(hyperparam_list)
train_mse = [0]*len(hyperparam_list)
train_err = [0]*len(hyperparam_list)
results = [0]*len(hyperparam_list)

print("\nRandom search script started for the",network_type,"\n")
expt_data, raw_testing_data, scaler_test = data_preprocess(network_type)
init_time = datetime.now()
if parallel_type == 'configurations':
    results[:] = Parallel(n_jobs=cores)(delayed(neural_network_evaluation)(cfg,
                    expt_data, cores, n_samples, network_type, parallel_type)  for cfg in hyperparam_list)
    print_results(hyperparam_list, results, network_type) # in function prints are suppressed by joblib 
elif parallel_type == 'samples':
    for j in range(0,len(hyperparam_list)):
        results[j] = neural_network_evaluation(hyperparam_list[j], expt_data, cores, n_samples, network_type, parallel_type)
else:
    print("Invalid parallel processing method, choose either 'samples' or 'configurations'.")
    exit(1)
        
for j in range(0,len(hyperparam_list)):
    avg_mse_data[j] = results[j][0]
    mse_err[j]      = results[j][1]
    train_mse[j]    = results[j][2]
    train_err[j]    = results[j][3]
fin_time = datetime.now()
print("\nRandom search algorithm execution time:", (fin_time-init_time), "(hrs:mins:secs)")
print("No. of configurations:", len(hyperparam_list)," samples:",n_samples," cores:",cores,"\n" )
        
# Outputting the best 8 hyperparameter sets
sorted_indexes = np.argsort(avg_mse_data) # Returns a list of original indexes if the array were to be sorted
if len(avg_mse_data) < 8:
    minimum_vals = len(avg_mse_data)
else:
    minimum_vals = 8
    
for i in range (0, minimum_vals):
    min_index = sorted_indexes[i]
    cfg          = hyperparam_list[min_index]
    avg_mse      = results[min_index][0]
    std_mse      = results[min_index][1]
    train_mse    = results[min_index][2]
    train_std    = results[min_index][3]
    if network_type == 'fnn':
        print('%d) HN (%d, %d), Epochs %d, BS %d, LR %.6f, MSE (%.4f \u00b1 %.4f), Train MSE (%.4f \u00b1 %.4f)'% (i+1,
                                            cfg[0],cfg[1],cfg[2],cfg[3],cfg[4], avg_mse, std_mse, train_mse, train_std))
        hyperparam_names = ["MSE","std","Train MSE","Train std","H1","H2","Epochs", "Batch Size", "Learning rate"]
    if network_type == 'rnn':
        print('%d) HN %d, Epochs %d, BS %d, LR %.6f, MSE (%.4f \u00b1 %.4f), Train MSE (%.4f \u00b1 %.4f)'% (i+1,
                                            cfg[0],cfg[1],cfg[2],cfg[3], avg_mse, std_mse, train_mse, train_std))
        hyperparam_names = ["MSE","std","Train MSE","Train std","HN","Epochs", "Batch Size", "Learning rate"]
    
filename = 'random_test.csv'
write_to_csv(filename, results, hyperparam_list, hyperparam_names, n_samples, fin_time-init_time, cores)
