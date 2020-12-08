# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 19:47:11 2020
Author: Adam Coxson, MPhys Undergrad, The University of Manchester
Project: Evaluating hyperparameter optimisation techinques for Bioprocesses
"Finding the Optimal Optimiser"
Module: HPO_random_and_grid_search 

PLAN #######################################################################
Hyperparmeters to consider
No. neurons per layer (assume the same no. for all layers, no combinations)
No. hidden layers
Activation functions in each layer - Sinh, Tanh, ReLU
Learning rate - the magnitude to update weights by during back propagation
Batch size - The number of datapoints to process before weights update
No. of EPOCHs - The number of iterations through the whole dataset 
EPOCHs are the number of network iterations required to train the network. 
These will either be inputted directly by the programmer, continue until a
preset time limit, continue until a minimum error is achieved.

This code is my initial attempt at automating grid and random search for a 
specific neural network.
Tutorial references
https://github.com/aggarwalpiush/Hyperparameter-Optimization-Tutorial
https://towardsdatascience.com/random-search-vs-grid-search-for-hyperparameter-optimization-345e1422899d

Summary 28/06/20
I think I've got the code setup to iterate through every single combination, however, it seems to go on
indefinitely without stopping and I'm not sure why. It could be a looping error, it could be due to 
trying all combinations in a single set of nested loops such that the dimensionality increases dramatically.

See results of plotting in "my code\Hyperparameter optimisation\graphs\Testing 28JUN20"
 Epochs and LR are more signifcant


"""

import numpy as np                         # Packages -----
import random
import copy
import csv
from scipy.stats import loguniform
from cpuinfo import get_cpu_info
from datetime import datetime
#from joblib.externals.loky import set_loky_pickler
from joblib import Parallel, delayed
#from joblib import wrap_non_picklable_objects
from rnn import RNN                            # Modules -----
from train import train
from test_module import test
from data_preprocessing import data_preprocess


def neural_net(cfg):
    HN    = int(cfg[0])
    EPOCHS = int(cfg[1])
    bs     = int(cfg[2])
    lr     = float(cfg[3])
    
    HL = 1 # one layer RNN
    net = RNN(1, 4, 10, HN, HL)
    init_state = copy.deepcopy(net.state_dict())
    net.load_state_dict(init_state)
    train(net, training_inputs, training_labels, EPOCHS, lr, bs)
    avg_mse = float(test(test_inputs, test_labels, net))
    return avg_mse

def write_to_csv(filename, mse_values, x_iters, hyperparameter_names, time, cores):
    
    for i in range(0,len(mse_values)):
        x_iters[i].insert(0,mse_values[i])
    hyperparameter_names.insert(0,"MSE")
    
    writer = csv.writer(open(filename,'w'),lineterminator ='\n')
    writer.writerow(["Time:", time, "Size:", len(mse_values),"Cores:",cores,"CPU:", get_cpu_info()['brand_raw']])
    writer.writerow(hyperparameter_names)
    writer.writerows(x_iters)
    print("Written to",filename,"successfully.")


training_data, testing_data = data_preprocess()

training_inputs = training_data[:, 0:4]
training_labels = training_data[:, 4:]
test_inputs = testing_data[:, 0:4]
test_labels = testing_data[:, 4:]

training_inputs = np.split(training_inputs, 820)
training_labels = np.split(training_labels, 820)
test_inputs = np.split(test_inputs, 5)
test_labels = np.split(test_labels, 5)

# Random Search Training Loop
# 2 Hidden Layers, tanh activation function for Recurrent neural network

combinations = 20 # number of different hyperparameter configurations
hyperparam_list = []
avg_mse_data = [0]*combinations


for i in range(0,combinations):
    hn = np.random.randint(low=1, high=20)
    epochs = np.random.randint(low=10, high=400)
    bs = np.random.randint(low=1, high=300)
    lr = round(loguniform.rvs(10**-4, 1*(10**-1)),5)
    hyperparam_list.append([hn, epochs, bs, lr])

do_parallel = True
cores = 2 # No. of cores for parallel processing
init_time = datetime.now()

if do_parallel == True:
    print("Parallel processing activated, print functions are surpressed.")
    #with parallel_backend("loky", inner_max_num_threads=2):
    avg_mse_data[:] = Parallel(n_jobs=cores)(delayed(neural_net)(cfg) for cfg in hyperparam_list)
else: 
    i = 0    
    for cfg in hyperparam_list:
        avg_mse_data[hyperparam_list.index(cfg)] = neural_net(cfg)
        i = i + 1
        print('%d) Parameters HN: %d,  Epochs: %d,  Batch Size: %d,  Learn rate: %.4f'%(i,cfg[0],cfg[1],cfg[2],cfg[3]))
    
fin_time = datetime.now()
print("Random search algorithm execution time: ", (fin_time-init_time), "(hrs:mins:secs)")
print("No. of parameter configurations: ", len(hyperparam_list) )
        
# OUTPUTS # ------------------------------------
sorted_indexes = np.argsort(avg_mse_data) # Returns a list of original indexes if the array were to be sorted
for i in range (0,8):
    min_index = sorted_indexes[i]
    cfg = hyperparam_list[min_index]
    print('%d) Parameters HN: %d, Epochs: %d, Batch Size: %d, Learn rate: %.4f'%(i+1,cfg[0],cfg[1],cfg[2],cfg[3]))
    print("MSE: ", np.round(avg_mse_data[min_index],5), "\n")
filename = 'RandSearch_RNN_test.csv'
write_to_csv(filename, avg_mse_data, hyperparam_list, ["HN","Epochs", "Learing rate", "Batch size"], fin_time-init_time,cores)
