# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 19:47:11 2020
Author: Adam Coxson, MPhys Undergrad, The University of Manchester
Project: Evaluating hyperparameter optimisation techinques for Bioprocesses
"Finding the Optimal Optimiser"
Module: HPO_random_and_grid_search_B

The _B identifier indicates it uses Dongda's experimental data 

PLAN #######################################################################
Hyperparmeters to consider
No. neurons per layer (assume the same no. for all layers, no combinations)
No. hidden layers
Activation functions in each layer - Sinh, Tanh, ReLU
Learning rate - the magnitude to update weights by during back propagation
Batch size - The number of datapoints to process before weights update
No. of EPOCHs - The number of iterations through the whole dataset 
EPOCHs are the number of network iterations required to train the network. 

This script conducts grid search of a given neural network from ann_B.py.
From performing 3 sets of bayesian optimisation on this data, I have identifed
a Hidden network neuron number of (4,2) to consistently give some of the 
lowest MSE values. As such I will use (4,2) with sigmoid activations and vary
Learning raet, Epoch number and batch size.
"""

import numpy as np                   # Packages -----
import copy
import csv
from cpuinfo import get_cpu_info
from datetime import datetime
#from joblib.externals.loky import set_loky_pickler
from joblib import Parallel, delayed
#from joblib import wrap_non_picklable_objects
from ann import Net                # Modules -----
from train_v2 import train
from test_module import test
from data_preprocessing import data_preprocess


def neural_net(cfg):
    h1     = int(cfg[0])
    h2     = int(cfg[1])
    EPOCHS = int(cfg[2])
    bs     = int(cfg[3])
    lr     = float(cfg[4])
    
    net = Net(h1, h2)
    init_state = copy.deepcopy(net.state_dict())
    net.load_state_dict(init_state)
    train(net, training_inputs, training_labels, EPOCHS, lr, bs)
    avg_mse = test(test_inputs, test_labels, net)
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

training_data, testing_data, raw_testing_data, scaler_test = data_preprocess()

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

EPOCHS = [40, 80, 120]
LR = [0.001, 0.005, 0.01]
HN = [(2, 4),(4,2)]
batch_size = [100]

training_inputs = training_data[:, 0:4]
training_labels = training_data[:, 4:]
test_inputs = testing_data[:, 0:4]
test_labels = testing_data[:, 4:]
hyperparam_list = []
avg_mse_data = [0]*len(HN)*len(EPOCHS)*len(batch_size)*len(LR)




for h in HN:
    for epochs in EPOCHS:
        for bs in batch_size:
            for lr in LR:
                hyperparam_list.append([h[0],h[1],epochs,bs,lr])

# for epochs in EPOCHS:
#     for bs in batch_size:
#         hyperparam_list.append([8,4,epochs,bs,0.01])
                

do_parallel = False
cores = 4
init_time = datetime.now()

if do_parallel == True:
    print("Parallel processing activated, print functions are surpressed.")
    avg_mse_data[:] = Parallel(n_jobs=cores)(delayed(neural_net)(config) for config in hyperparam_list)
else: 
    i = 0    
    for cfg in hyperparam_list:
        avg_mse_data[hyperparam_list.index(cfg)] = neural_net(cfg)
        i = i + 1
        print('%d) Parameters HN: (%d, %d), Epochs: %d, Batch Size: %d, Learn rate: %.4f'%(i,cfg[0],cfg[1],cfg[2],cfg[3],cfg[4]))

fin_time = datetime.now()
print("Grid search algorithm execution time: ", (fin_time-init_time), "(hrs:mins:secs)")
print("No. of parameter configurations: ", len(hyperparam_list) )
        
# OUTPUTS # ------------------------------------
sorted_indexes = np.argsort(avg_mse_data) # Returns a list of original indexes if the array were to be sorted
for i in range (0,8):
    min_index = sorted_indexes[i]
    cfg = hyperparam_list[min_index]
    print('%d) Parameters HN: (%d, %d), Epochs: %d, Batch Size: %d, Learn rate: %.4f'%(i+1,cfg[0],cfg[1],cfg[2],cfg[3],cfg[4]))
    print("MSE: ", np.round(avg_mse_data[min_index],5), "\n")
filename = 'Grid_test.csv'
write_to_csv(filename, avg_mse_data, hyperparam_list, ["H1","H2","Epochs","Batch size","Learning rate"],fin_time-init_time, cores)


