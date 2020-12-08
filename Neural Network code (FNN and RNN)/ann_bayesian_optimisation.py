# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 16:54:01 2020
Author: Adam Coxson, MPhys Undergrad, The University of Manchester
Project: Evaluating hyperparameter optimisation techinques for an FNN and RNN
Module: ann_bayesian_optimzation  (Gaussian Process Bayesian Optimisation)
Dependancies: neural_networks, network_train_and_test_module, training_function, testing_functions, data_preprocessing

This script performs Gaussian Process Bayesian Optimisation for either an FNN or RNN neural network architecture.
The results are written to an excel file, execution times (for 250 evaluations) can last from 1 hour to 12 hours.
Note that parallel processing is used to evaluate the same set of hyperparameters several times (n_samples), this
ensures that the outputted accuracy reading from the neural network is representative of the parameters. The mean 
of these is taken and half the standard deviation is also added on, this helps to bias the output so gp_minimize 
selects parameters with small deviations from their mean.

gp_minimize is from the skopt.optimze package which also contain a random forest algorithm, forest_minimize and 
for random search there is is dummy_minimize. Note, to ensure use_named_args works, I have had to place the line
 @use_named_args(bounds) in front of the definitions for F/RNN_evaluation().

For random forest, replace gp_minimize() with:
rf_output = forest_minimize(func,           
                     dimensions=bounds,    
                     acq_func="EI",        
                     n_calls = num_evaluations,          
                     n_random_starts = random_vals,                         
                     random_state=1234)     

Resources:
https://scikit-optimize.github.io/stable/modules/generated/skopt.gp_minimize.html
https://scikit-optimize.github.io/stable/auto_examples/strategy-comparison.html#sphx-glr-auto-examples-strategy-comparison-py
https://machinelearningmastery.com/what-is-bayesian-optimization/

"""
# Packages
import copy
import numpy as np
import warnings
from skopt.space import Integer, Real, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from datetime import datetime
from joblib import Parallel, delayed
# Modules 
from neural_networks import fnn_net, rnn_net             
from network_train_and_test_module import train_and_test_mse_only, write_to_csv
from data_preprocessing import data_preprocess

    
def neural_network_evaluation(cfg, expt_data, cores, n_samples, network_type):
    """
    ARGS: cfg: A single configuration of hyperparameters (Neuron config, Epochs, batch size, learning rate),
          expt_data: formatted and normalised training and testing data  
          cores: number of cores to be used in parallel processing,
          n_samples: number of samples to average over for a given set of hyperparameters,
          network_type: either 'fnn' or 'rnn'.
          
    OUTPUTS: avg_mse: average of all the different, n_sample test mse values, 
             std_mse: spread of all the different, n_sample test mse values, 
             train_mse: average of all the different, n_sample training mse values,  
             train_std: spread of all the different, n_sample training mse values, 
             mse_vals: list of all the testing mse value (each an average of the test sets).

    
    This function takes in a set of hyperparameters and evaluates them for the FNN or RNN architectures developed in the 
    program. It repeats the network training and testing n_samples times over and averages this data, utilising parallel
    processing via the train_and_test() function. Averaging over n_samples is necessary to give the Bayesian algorithm an 
    accurate estimate, otherwise single iterations of train and test have high variance so the Bayesian search struggles
    to find a final, good set of hyperparameters. For final values of mean and standard deviation for a given 
    hyperparameter set, it is recommended to use 10, 12, 16 or 20 samples. Bayesian optimisation does not use multiprocessing
    however, the joblib parallel package is used to speed up the training and testing loop over n_samples. It is useful to
    for n_samples to be a multiple of, or equal to, the number of cores.
    """

    if network_type == 'fnn': 
        H1     = int(cfg[0])
        H2     = int(cfg[1])
        EPOCHS = int(cfg[2])
        bs     = int(cfg[3])
        lr     = float(cfg[4])
    elif network_type == 'rnn':
        HL = 1
        HN    = int(cfg[0])
        EPOCHS = int(cfg[1])
        bs     = int(cfg[2])
        lr     = float(cfg[3])
    else:
        print("Invalid neural network type, choose either fnn or rnn")
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
    # Joblib parallelisation across the number of samples
    results[:] = Parallel(n_jobs=cores)(delayed(train_and_test_mse_only)(EPOCHS, lr, bs, expt_data, network_type, net) for net in networks)
    for i in range (0,n_samples):
        mse_vals[i] = results[i][0]
        training_mse_vals[i] = np.array(float(results[i][1]))
        
    avg_mse = np.mean(mse_vals)
    std_mse = np.std(mse_vals)
    train_mse = np.mean(training_mse_vals) 
    train_std = np.std(training_mse_vals)
    if network_type == 'fnn':
        print('HN (%d, %d), Epochs %d, BS %d, LR %.6f, GP_output %.4f, MSE (%.4f \u00b1 %.4f), Train MSE (%.4f \u00b1 %.4f)'% (H1, H2,
                                                EPOCHS, bs, lr,avg_mse+(std_mse/2), avg_mse, std_mse, train_mse, train_std))
    if network_type == 'rnn':
        print('HN %d, Epochs %d, BS %d, LR %.6f, GP_output %.4f, MSE (%.4f \u00b1 %.4f), Train MSE (%.4f \u00b1 %.4f)'% (HN,
                                                EPOCHS, bs, lr, avg_mse+(std_mse/2), avg_mse, std_mse, train_mse, train_std))
    return avg_mse, std_mse, train_mse, train_std

# # # # # # MAIN # # # # # #

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
"""
This filters the following warnings (tbh i don't really know what the first one means):
C:\ProgramData\Anaconda3\lib\site-packages\torch\storage.py:34: FutureWarning: pickle support for Storage will be removed in 1.5. Use `torch.save` instead
warnings.warn("pickle support for Storage will be removed in 1.5. Use `torch.save` instead", FutureWarning)
C:\ProgramData\Anaconda3\lib\site-packages\joblib\externals\loky\process_executor.py:691: UserWarning: A worker 
stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak
"""

random_vals = 50
num_evaluations = random_vals + 200
random_vals = 10                 # For debug
num_evaluations = random_vals + 20
cores = 4
n_samples = 4  # number of iterations for training and testing - recommend at least 12
all_config_results = [0]*num_evaluations
hyperparam_idx = 0
network_type = 'fnn'

expt_data, raw_testing_data, test_scaler = data_preprocess(network_type) 

if network_type == 'fnn':
    print("\nScript Started for the FNN.\n")
    # bounds = [Categorical([2,4,6,8,10], name = 'H1'),
    #           Categorical([2,4], name = 'H2'),
    #           Integer(200, 800,  name='EPOCHS'), 
    #           #Integer(1, 400,  name='BS'),
    #           Categorical([100,200,300,400], name = 'BS'),
    #           Real(0.0001,0.02,"log-uniform",  name='LR')]
    bounds = [                               # A quick set for debug
          Categorical([2,4], name = 'H1'),
          Categorical([2,4], name = 'H2'),
          Integer(10, 50,  name='EPOCHS'), 
          Categorical([200,250,300,350,400], name = 'BS'),
          Real(0.0001,0.02,"log-uniform",  name='LR')]
elif network_type == 'rnn':
    print("\nScript Started for the RNN.\n")
    bounds = [Integer(1, 30,  name='HN'),
              Integer(1, 200,  name='EPOCHS'), 
              Integer(1, 600,  name='BS'),
              #Categorical([20, 50, 80, 100,150,200,300,400,500], name = 'BS'),
              Real(0.0001,0.02,"log-uniform",  name='LR')]
else:
    print("Invalid neural network type, choose either fnn or rnn")
          
@use_named_args(bounds)
def FNN_evaluation(H1,H2,EPOCHS,BS,LR):
    """
    This function is defined here to be compatible with gp_minimize, below the 'bounds' variable and @use_named_args.
    A single output is used to pass straight into the Bayesian optimisation. 
    """
    cfg = [H1, H2, EPOCHS, BS, LR]
    global hyperparam_idx
    avg_mse, std_mse, train_mse, train_std = neural_network_evaluation(cfg, expt_data, cores, n_samples, 'fnn')
    all_config_results[hyperparam_idx] = [avg_mse, std_mse, train_mse, train_std] # Getting all the averaged data
    hyperparam_idx = hyperparam_idx + 1
    return avg_mse + std_mse/2 # Return the average mse plus some of its error. Biases gp_minimize to use smaller stds

@use_named_args(bounds)
def RNN_evaluation(HN,EPOCHS,BS,LR):
    """
    This function is defined here to be compatible with gp_minimize, below the 'bounds' variable and @use_named_args.
    A single output is used to pass straight into the Bayesian optimisation. 
    """
    global hyperparam_idx
    cfg = [HN, EPOCHS, BS, LR]
    avg_mse, std_mse, train_mse, train_std = neural_network_evaluation(cfg, expt_data, cores, n_samples, 'rnn')
    all_config_results[hyperparam_idx] = [avg_mse, std_mse, train_mse, train_std] # Getting all the averaged data
    hyperparam_idx = hyperparam_idx + 1
    return avg_mse + std_mse/2 # Return the average mse plus some of its error. Biases gp_minimize to use smaller stds
    
if network_type == 'fnn':
    func = FNN_evaluation
else:
    func = RNN_evaluation
    
# GAUSSIAN PROCESS BAYESIAN OPTIMISATION TUNING PROCEDURE
init_time = datetime.now()
gp_output = gp_minimize(func,                       # function to minimize
                     dimensions=bounds,             # bounds on each hyperparameter dimension 
                     acq_func="EI",                 # acquisition function (Expected improvement)
                     n_calls = num_evaluations,     # number of evaluations of func
                     n_random_starts = random_vals, # number of initial, random seeding points                       
                     random_state=1234)             # some random seed
fin_time = datetime.now()
print("\nBayesian GP Optimisation execution time: ", (fin_time-init_time), "(hrs:mins:secs)")

if network_type == 'fnn': # Optimal result output
    print('Results: MSE: %.4f, HN: (%d, %d), Epochs: %d, Batch Size: %d, Learn rate: %.6f' % (gp_output.fun,
                              gp_output.x[0], gp_output.x[1],gp_output.x[2], gp_output.x[3],gp_output.x[4]))
    hyperparams = ["MSE","std","Train MSE","Train std","H1","H2","Epochs", "Batch Size", "Learning rate"]
    
if network_type =='rnn':
    print('Results: MSE: %.4f, HN: %d, Epochs %d, Batch Size %d, Learn rate %.6f' % (gp_output.fun,
                                    gp_output.x[0], gp_output.x[1], gp_output.x[2], gp_output.x[3]))
    hyperparams = ["MSE","std","Train MSE","Train std","HN","Epochs", "Batch Size", "Learning rate"]
    
filename = 'Bayesian_test'+'.csv'
write_to_csv(filename,all_config_results, gp_output.x_iters, hyperparams,n_samples,fin_time-init_time, cores)



    