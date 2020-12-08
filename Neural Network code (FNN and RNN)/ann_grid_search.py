# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 16:54:01 2020
Author: Adam Coxson, MPhys Undergrad, The University of Manchester
Project: Evaluating hyperparameter optimisation techinques for an FNN and RNN
Module: ann_grid_search
Dependancies: neural_networks, network_train_and_test_module, training_function, testing_functions, data_preprocessing

This script performs grid search upon a defined list of hyperparameters for either an FNN or an RNN. Parallel 
processing is used to speed up the process. Two options for this are offered:
The 'samples' method means a each configuration is evaluated in turn and the multiprocessing is applied at the 
training and testing loop, where the network is reset and evaluated n_sample times over.
The 'configurations' method uses multiprocessing to evaluate multiple configurations of hyperparameters at the
same time. This is useful for computer clusters and having access to many more cores than n_samples.
Evaluating each set of hyperparameters several times over (n_samples) ensures that the outputted 
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
# Grid Search 

cores = 4
n_samples = 10
network_type = 'rnn'
parallel_type = 'samples'
#parallel_type = 'configurations'

# HN = [(2,2),(4,2),(6,2),(8,2),(10,2), (2,4), (4,4), (6,4), (8,4), (10,4)] # size 10
# EPOCHS = [ 10,  25, 50,  100, 150, 200, 250, 300, 350, 400] #size 10
# batch_size = [ 1, 10, 25, 50, 80, 100, 150, 200, 250, 300] # size 10
# LR = [0.0001, 0.0002, 0.0004, 0.001, 0.002, 0.004, 0.01, 0.02, 0.04, 0.1] # size 10


#EPOCHS = np.linspace(20,1000,99)
#EPOCHS = [10,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800]
EPOCHS = [20, 50, 100]
LR = [0.001]
#LR = np.round(np.logspace(-4, -3, num = 30),6)
#LR = [0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
HN = [(8,4),(4,2)]
#HN = [2,4]
batch_size = [100,200]
#batch_size = np.linspace(50,800,16)
#batch_size[0] = 1

hyperparam_list = []
for h in HN:
    for epochs in EPOCHS:
        for bs in batch_size:
            for lr in LR:
                if network_type == 'fnn':
                    hyperparam_list.append([h[0],h[1],epochs,bs,lr])
                elif network_type == 'rnn':
                    hyperparam_list.append([h,epochs,bs,lr])

# Some optimal configs for the FNN ~ 1 hour to excecute all 5
#hyperparam_list = [[4,4,800,100,0.0001],[8,4,186,20,0.0001],[8,4,141,456,0.0010],[8,4,245,55,0.00015],[8,4,510,200,0.00015]]
# Some optimal configs for RNN ~ 20 minutes to run all 4
#hyperparam_list = [[10,50,121,0.002913],[23,96,103,0.000762],[25,36,1,0.000106],[30,141,77,0.000443]]

avg_mse_data = [0]*len(hyperparam_list)
mse_err = [0]*len(hyperparam_list)
train_mse = [0]*len(hyperparam_list)
train_err = [0]*len(hyperparam_list)
results = [0]*len(hyperparam_list)

print("\nGrid search script started for the",network_type,"\n")
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
print("\nGrid search algorithm execution time:", (fin_time-init_time), "(hrs:mins:secs)")
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
    
filename = 'Grid_test.csv'
write_to_csv(filename, results, hyperparam_list, hyperparam_names, n_samples, fin_time-init_time, cores)


