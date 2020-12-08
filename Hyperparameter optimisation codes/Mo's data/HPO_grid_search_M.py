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

import pandas as pd
import numpy as np 
import copy
from ann2 import Net
from replicate import replicate_data 
from sklearn.preprocessing import StandardScaler
from train2 import train
from test2 import test
from datetime import datetime
from data_preprocessing import data_preprocess
import csv

def write_to_csv(filename, mse_values, x_iters, hyperparameter_names,time):
    
    for i in range(0,len(mse_values)):
        x_iters[i].insert(0,mse_values[i])
    hyperparameter_names.insert(0,"MSE")
    
    writer = csv.writer(open(filename,'w'),lineterminator ='\n')
    writer.writerow(["Time:", time])
    writer.writerow(hyperparameter_names)
    writer.writerows(x_iters)
    print("Written to",filename,"successfully.")

training_data, testing_data = data_preprocess()

# Grid Search Training Loop
HL = 2
HN = [4,8]
EPOCHS = np.linspace(0,500,51)
#BATCH_SIZE = np.linspace(0,400, 41)

# EPOCHS = [50, 100, 150]
# BATCH_SIZE = [50, 100, 150]
#LR = 0.006
#LR = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25]
LR = np.logspace(-4, -1, num = 49)
LR = np.round(LR[:],6)

EPOCHS[0] = 1

MODELS = {}
training_inputs = training_data[:, 0:5]
training_labels = training_data[:, 5:]
test_inputs = testing_data[:, 0:5]
test_labels = testing_data[:, 5:]
hyperparam_config_list = []
#avg_mse_data = [0]*len(HN)*len(EPOCHS)*len(BATCH_SIZE)*len(LR)
avg_mse_data = [0]*len(EPOCHS)*len(LR)

# for h in HN:
#     for epochs in EPOCHS:
#         for lr in LR:
#             for bs in BATCH_SIZE:
#                 hyperparam_config_list.append([h,epochs,lr,bs])

for epochs in EPOCHS:
    for lr in LR:
        hyperparam_config_list.append([epochs,lr])
init_time = datetime.now()

for config in hyperparam_config_list:
    lr = float(config[1])
    bs = 50
    EPOCHS = int(config[0])
    #lr = 0.006
    h1 = 4
    h2 = 8
    
    net = Net(4, 8)
    init_state = copy.deepcopy(net.state_dict())
    net.load_state_dict(init_state)
    train(net, training_inputs, training_labels, EPOCHS, lr, bs)
    avg_mse = test(test_inputs, test_labels, net)
    avg_mse_data[hyperparam_config_list.index(config)] = avg_mse[0]
    
    MODELS['{a}_{x}-{y}_{z}_{b}_{c}'.format(a=HL, x=h1, y=h2, z=EPOCHS, b=lr, c=bs)] = avg_mse

with open('Data3/initial_test_results/test_results_{x}HL_bsTIMETEST.csv'.format(x=HL), 'w') as f:
    for key in MODELS.keys():
        f.write("%s: %s\n"%(key, MODELS[key]))

fin_time = datetime.now()
print("Grid search algorithm execution time: ", (fin_time-init_time), "(hrs:mins:secs)")
print("No. of parameter configurations: ", len(hyperparam_config_list) )
        
# OUTPUTS # ------------------------------------
sorted_indexes = np.argsort(avg_mse_data) # Returns a list of original indexes if the array were to be sorted
for i in range (0,8):
    min_index = sorted_indexes[i]
    config = hyperparam_config_list[min_index]
    #print(i+1,") Params: HN = ", config[0], " Epochs = ", config[1]," LR = ", config[2]," Batch Size = ", config[3])
    print(i+1,") Params: HN = (4,8), Epochs = ", config[0]," LR = ", config[1]," Batch Size = 50")
    print("MSE: ", avg_mse_data[min_index], "\n")
filename = 'Grid_search_LR_EPOCHS.csv'
write_to_csv(filename, avg_mse_data, hyperparam_config_list, ["Epochs", "Learning Rate"], fin_time-init_time)
"""
#TESTING/DEMO
# Sorting and MSE output Testing
# data_store is saved MSE data from a previous run of 500 configurations
hyperparam_list_test = [[2, 50, 0.002, 20], [2, 50, 0.002, 50], [2, 50, 0.002, 100], [2, 50, 0.002, 200], [2, 50, 0.002, 300], [2, 50, 0.004, 20], [2, 50, 0.004, 50], [2, 50, 0.004, 100], [2, 50, 0.004, 200], [2, 50, 0.004, 300], [2, 50, 0.006, 20], [2, 50, 0.006, 50], [2, 50, 0.006, 100], [2, 50, 0.006, 200], [2, 50, 0.006, 300], [2, 50, 0.008, 20], [2, 50, 0.008, 50], [2, 50, 0.008, 100], [2, 50, 0.008, 200], [2, 50, 0.008, 300], [2, 50, 0.01, 20], [2, 50, 0.01, 50], [2, 50, 0.01, 100], [2, 50, 0.01, 200], [2, 50, 0.01, 300], [2, 100, 0.002, 20], [2, 100, 0.002, 50], [2, 100, 0.002, 100], [2, 100, 0.002, 200], [2, 100, 0.002, 300], [2, 100, 0.004, 20], [2, 100, 0.004, 50], [2, 100, 0.004, 100], [2, 100, 0.004, 200], [2, 100, 0.004, 300], [2, 100, 0.006, 20], [2, 100, 0.006, 50], [2, 100, 0.006, 100], [2, 100, 0.006, 200], [2, 100, 0.006, 300], [2, 100, 0.008, 20], [2, 100, 0.008, 50], [2, 100, 0.008, 100], [2, 100, 0.008, 200], [2, 100, 0.008, 300], [2, 100, 0.01, 20], [2, 100, 0.01, 50], [2, 100, 0.01, 100], [2, 100, 0.01, 200], [2, 100, 0.01, 300], [2, 200, 0.002, 20], [2, 200, 0.002, 50], [2, 200, 0.002, 100], [2, 200, 0.002, 200], [2, 200, 0.002, 300], [2, 200, 0.004, 20], [2, 200, 0.004, 50], [2, 200, 0.004, 100], [2, 200, 0.004, 200], [2, 200, 0.004, 300], [2, 200, 0.006, 20], [2, 200, 0.006, 50], [2, 200, 0.006, 100], [2, 200, 0.006, 200], [2, 200, 0.006, 300], [2, 200, 0.008, 20], [2, 200, 0.008, 50], [2, 200, 0.008, 100], [2, 200, 0.008, 200], [2, 200, 0.008, 300], [2, 200, 0.01, 20], [2, 200, 0.01, 50], [2, 200, 0.01, 100], [2, 200, 0.01, 200], [2, 200, 0.01, 300], [2, 300, 0.002, 20], [2, 300, 0.002, 50], [2, 300, 0.002, 100], [2, 300, 0.002, 200], [2, 300, 0.002, 300], [2, 300, 0.004, 20], [2, 300, 0.004, 50], [2, 300, 0.004, 100], [2, 300, 0.004, 200], [2, 300, 0.004, 300], [2, 300, 0.006, 20], [2, 300, 0.006, 50], [2, 300, 0.006, 100], [2, 300, 0.006, 200], [2, 300, 0.006, 300], [2, 300, 0.008, 20], [2, 300, 0.008, 50], [2, 300, 0.008, 100], [2, 300, 0.008, 200], [2, 300, 0.008, 300], [2, 300, 0.01, 20], [2, 300, 0.01, 50], [2, 300, 0.01, 100], [2, 300, 0.01, 200], [2, 300, 0.01, 300], [2, 400, 0.002, 20], [2, 400, 0.002, 50], [2, 400, 0.002, 100], [2, 400, 0.002, 200], [2, 400, 0.002, 300], [2, 400, 0.004, 20], [2, 400, 0.004, 50], [2, 400, 0.004, 100], [2, 400, 0.004, 200], [2, 400, 0.004, 300], [2, 400, 0.006, 20], [2, 400, 0.006, 50], [2, 400, 0.006, 100], [2, 400, 0.006, 200], [2, 400, 0.006, 300], [2, 400, 0.008, 20], [2, 400, 0.008, 50], [2, 400, 0.008, 100], [2, 400, 0.008, 200], [2, 400, 0.008, 300], [2, 400, 0.01, 20], [2, 400, 0.01, 50], [2, 400, 0.01, 100], [2, 400, 0.01, 200], [2, 400, 0.01, 300], [4, 50, 0.002, 20], [4, 50, 0.002, 50], [4, 50, 0.002, 100], [4, 50, 0.002, 200], [4, 50, 0.002, 300], [4, 50, 0.004, 20], [4, 50, 0.004, 50], [4, 50, 0.004, 100], [4, 50, 0.004, 200], [4, 50, 0.004, 300], [4, 50, 0.006, 20], [4, 50, 0.006, 50], [4, 50, 0.006, 100], [4, 50, 0.006, 200], [4, 50, 0.006, 300], [4, 50, 0.008, 20], [4, 50, 0.008, 50], [4, 50, 0.008, 100], [4, 50, 0.008, 200], [4, 50, 0.008, 300], [4, 50, 0.01, 20], [4, 50, 0.01, 50], [4, 50, 0.01, 100], [4, 50, 0.01, 200], [4, 50, 0.01, 300], [4, 100, 0.002, 20], [4, 100, 0.002, 50], [4, 100, 0.002, 100], [4, 100, 0.002, 200], [4, 100, 0.002, 300], [4, 100, 0.004, 20], [4, 100, 0.004, 50], [4, 100, 0.004, 100], [4, 100, 0.004, 200], [4, 100, 0.004, 300], [4, 100, 0.006, 20], [4, 100, 0.006, 50], [4, 100, 0.006, 100], [4, 100, 0.006, 200], [4, 100, 0.006, 300], [4, 100, 0.008, 20], [4, 100, 0.008, 50], [4, 100, 0.008, 100], [4, 100, 0.008, 200], [4, 100, 0.008, 300], [4, 100, 0.01, 20], [4, 100, 0.01, 50], [4, 100, 0.01, 100], [4, 100, 0.01, 200], [4, 100, 0.01, 300], [4, 200, 0.002, 20], [4, 200, 0.002, 50], [4, 200, 0.002, 100], [4, 200, 0.002, 200], [4, 200, 0.002, 300], [4, 200, 0.004, 20], [4, 200, 0.004, 50], [4, 200, 0.004, 100], [4, 200, 0.004, 200], [4, 200, 0.004, 300], [4, 200, 0.006, 20], [4, 200, 0.006, 50], [4, 200, 0.006, 100], [4, 200, 0.006, 200], [4, 200, 0.006, 300], [4, 200, 0.008, 20], [4, 200, 0.008, 50], [4, 200, 0.008, 100], [4, 200, 0.008, 200], [4, 200, 0.008, 300], [4, 200, 0.01, 20], [4, 200, 0.01, 50], [4, 200, 0.01, 100], [4, 200, 0.01, 200], [4, 200, 0.01, 300], [4, 300, 0.002, 20], [4, 300, 0.002, 50], [4, 300, 0.002, 100], [4, 300, 0.002, 200], [4, 300, 0.002, 300], [4, 300, 0.004, 20], [4, 300, 0.004, 50], [4, 300, 0.004, 100], [4, 300, 0.004, 200], [4, 300, 0.004, 300], [4, 300, 0.006, 20], [4, 300, 0.006, 50], [4, 300, 0.006, 100], [4, 300, 0.006, 200], [4, 300, 0.006, 300], [4, 300, 0.008, 20], [4, 300, 0.008, 50], [4, 300, 0.008, 100], [4, 300, 0.008, 200], [4, 300, 0.008, 300], [4, 300, 0.01, 20], [4, 300, 0.01, 50], [4, 300, 0.01, 100], [4, 300, 0.01, 200], [4, 300, 0.01, 300], [4, 400, 0.002, 20], [4, 400, 0.002, 50], [4, 400, 0.002, 100], [4, 400, 0.002, 200], [4, 400, 0.002, 300], [4, 400, 0.004, 20], [4, 400, 0.004, 50], [4, 400, 0.004, 100], [4, 400, 0.004, 200], [4, 400, 0.004, 300], [4, 400, 0.006, 20], [4, 400, 0.006, 50], [4, 400, 0.006, 100], [4, 400, 0.006, 200], [4, 400, 0.006, 300], [4, 400, 0.008, 20], [4, 400, 0.008, 50], [4, 400, 0.008, 100], [4, 400, 0.008, 200], [4, 400, 0.008, 300], [4, 400, 0.01, 20], [4, 400, 0.01, 50], [4, 400, 0.01, 100], [4, 400, 0.01, 200], [4, 400, 0.01, 300], [6, 50, 0.002, 20], [6, 50, 0.002, 50], [6, 50, 0.002, 100], [6, 50, 0.002, 200], [6, 50, 0.002, 300], [6, 50, 0.004, 20], [6, 50, 0.004, 50], [6, 50, 0.004, 100], [6, 50, 0.004, 200], [6, 50, 0.004, 300], [6, 50, 0.006, 20], [6, 50, 0.006, 50], [6, 50, 0.006, 100], [6, 50, 0.006, 200], [6, 50, 0.006, 300], [6, 50, 0.008, 20], [6, 50, 0.008, 50], [6, 50, 0.008, 100], [6, 50, 0.008, 200], [6, 50, 0.008, 300], [6, 50, 0.01, 20], [6, 50, 0.01, 50], [6, 50, 0.01, 100], [6, 50, 0.01, 200], [6, 50, 0.01, 300], [6, 100, 0.002, 20], [6, 100, 0.002, 50], [6, 100, 0.002, 100], [6, 100, 0.002, 200], [6, 100, 0.002, 300], [6, 100, 0.004, 20], [6, 100, 0.004, 50], [6, 100, 0.004, 100], [6, 100, 0.004, 200], [6, 100, 0.004, 300], [6, 100, 0.006, 20], [6, 100, 0.006, 50], [6, 100, 0.006, 100], [6, 100, 0.006, 200], [6, 100, 0.006, 300], [6, 100, 0.008, 20], [6, 100, 0.008, 50], [6, 100, 0.008, 100], [6, 100, 0.008, 200], [6, 100, 0.008, 300], [6, 100, 0.01, 20], [6, 100, 0.01, 50], [6, 100, 0.01, 100], [6, 100, 0.01, 200], [6, 100, 0.01, 300], [6, 200, 0.002, 20], [6, 200, 0.002, 50], [6, 200, 0.002, 100], [6, 200, 0.002, 200], [6, 200, 0.002, 300], [6, 200, 0.004, 20], [6, 200, 0.004, 50], [6, 200, 0.004, 100], [6, 200, 0.004, 200], [6, 200, 0.004, 300], [6, 200, 0.006, 20], [6, 200, 0.006, 50], [6, 200, 0.006, 100], [6, 200, 0.006, 200], [6, 200, 0.006, 300], [6, 200, 0.008, 20], [6, 200, 0.008, 50], [6, 200, 0.008, 100], [6, 200, 0.008, 200], [6, 200, 0.008, 300], [6, 200, 0.01, 20], [6, 200, 0.01, 50], [6, 200, 0.01, 100], [6, 200, 0.01, 200], [6, 200, 0.01, 300], [6, 300, 0.002, 20], [6, 300, 0.002, 50], [6, 300, 0.002, 100], [6, 300, 0.002, 200], [6, 300, 0.002, 300], [6, 300, 0.004, 20], [6, 300, 0.004, 50], [6, 300, 0.004, 100], [6, 300, 0.004, 200], [6, 300, 0.004, 300], [6, 300, 0.006, 20], [6, 300, 0.006, 50], [6, 300, 0.006, 100], [6, 300, 0.006, 200], [6, 300, 0.006, 300], [6, 300, 0.008, 20], [6, 300, 0.008, 50], [6, 300, 0.008, 100], [6, 300, 0.008, 200], [6, 300, 0.008, 300], [6, 300, 0.01, 20], [6, 300, 0.01, 50], [6, 300, 0.01, 100], [6, 300, 0.01, 200], [6, 300, 0.01, 300], [6, 400, 0.002, 20], [6, 400, 0.002, 50], [6, 400, 0.002, 100], [6, 400, 0.002, 200], [6, 400, 0.002, 300], [6, 400, 0.004, 20], [6, 400, 0.004, 50], [6, 400, 0.004, 100], [6, 400, 0.004, 200], [6, 400, 0.004, 300], [6, 400, 0.006, 20], [6, 400, 0.006, 50], [6, 400, 0.006, 100], [6, 400, 0.006, 200], [6, 400, 0.006, 300], [6, 400, 0.008, 20], [6, 400, 0.008, 50], [6, 400, 0.008, 100], [6, 400, 0.008, 200], [6, 400, 0.008, 300], [6, 400, 0.01, 20], [6, 400, 0.01, 50], [6, 400, 0.01, 100], [6, 400, 0.01, 200], [6, 400, 0.01, 300], [8, 50, 0.002, 20], [8, 50, 0.002, 50], [8, 50, 0.002, 100], [8, 50, 0.002, 200], [8, 50, 0.002, 300], [8, 50, 0.004, 20], [8, 50, 0.004, 50], [8, 50, 0.004, 100], [8, 50, 0.004, 200], [8, 50, 0.004, 300], [8, 50, 0.006, 20], [8, 50, 0.006, 50], [8, 50, 0.006, 100], [8, 50, 0.006, 200], [8, 50, 0.006, 300], [8, 50, 0.008, 20], [8, 50, 0.008, 50], [8, 50, 0.008, 100], [8, 50, 0.008, 200], [8, 50, 0.008, 300], [8, 50, 0.01, 20], [8, 50, 0.01, 50], [8, 50, 0.01, 100], [8, 50, 0.01, 200], [8, 50, 0.01, 300], [8, 100, 0.002, 20], [8, 100, 0.002, 50], [8, 100, 0.002, 100], [8, 100, 0.002, 200], [8, 100, 0.002, 300], [8, 100, 0.004, 20], [8, 100, 0.004, 50], [8, 100, 0.004, 100], [8, 100, 0.004, 200], [8, 100, 0.004, 300], [8, 100, 0.006, 20], [8, 100, 0.006, 50], [8, 100, 0.006, 100], [8, 100, 0.006, 200], [8, 100, 0.006, 300], [8, 100, 0.008, 20], [8, 100, 0.008, 50], [8, 100, 0.008, 100], [8, 100, 0.008, 200], [8, 100, 0.008, 300], [8, 100, 0.01, 20], [8, 100, 0.01, 50], [8, 100, 0.01, 100], [8, 100, 0.01, 200], [8, 100, 0.01, 300], [8, 200, 0.002, 20], [8, 200, 0.002, 50], [8, 200, 0.002, 100], [8, 200, 0.002, 200], [8, 200, 0.002, 300], [8, 200, 0.004, 20], [8, 200, 0.004, 50], [8, 200, 0.004, 100], [8, 200, 0.004, 200], [8, 200, 0.004, 300], [8, 200, 0.006, 20], [8, 200, 0.006, 50], [8, 200, 0.006, 100], [8, 200, 0.006, 200], [8, 200, 0.006, 300], [8, 200, 0.008, 20], [8, 200, 0.008, 50], [8, 200, 0.008, 100], [8, 200, 0.008, 200], [8, 200, 0.008, 300], [8, 200, 0.01, 20], [8, 200, 0.01, 50], [8, 200, 0.01, 100], [8, 200, 0.01, 200], [8, 200, 0.01, 300], [8, 300, 0.002, 20], [8, 300, 0.002, 50], [8, 300, 0.002, 100], [8, 300, 0.002, 200], [8, 300, 0.002, 300], [8, 300, 0.004, 20], [8, 300, 0.004, 50], [8, 300, 0.004, 100], [8, 300, 0.004, 200], [8, 300, 0.004, 300], [8, 300, 0.006, 20], [8, 300, 0.006, 50], [8, 300, 0.006, 100], [8, 300, 0.006, 200], [8, 300, 0.006, 300], [8, 300, 0.008, 20], [8, 300, 0.008, 50], [8, 300, 0.008, 100], [8, 300, 0.008, 200], [8, 300, 0.008, 300], [8, 300, 0.01, 20], [8, 300, 0.01, 50], [8, 300, 0.01, 100], [8, 300, 0.01, 200], [8, 300, 0.01, 300], [8, 400, 0.002, 20], [8, 400, 0.002, 50], [8, 400, 0.002, 100], [8, 400, 0.002, 200], [8, 400, 0.002, 300], [8, 400, 0.004, 20], [8, 400, 0.004, 50], [8, 400, 0.004, 100], [8, 400, 0.004, 200], [8, 400, 0.004, 300], [8, 400, 0.006, 20], [8, 400, 0.006, 50], [8, 400, 0.006, 100], [8, 400, 0.006, 200], [8, 400, 0.006, 300], [8, 400, 0.008, 20], [8, 400, 0.008, 50], [8, 400, 0.008, 100], [8, 400, 0.008, 200], [8, 400, 0.008, 300], [8, 400, 0.01, 20], [8, 400, 0.01, 50], [8, 400, 0.01, 100], [8, 400, 0.01, 200], [8, 400, 0.01, 300]]
data_store = [0.006941413835651136, 0.007792373634556988, 0.007732021260668637, 0.01933618683811675, 0.010178071672260132, 0.008421441365930654, 0.0072443516094214825, 0.0074071627264412085, 0.007858526238179672, 0.00862602222316392, 0.007099687775573135, 0.00620139637061853, 0.007161646064433394, 0.008582025341288494, 0.0073452861743454985, 0.006977138240770973, 0.007042068069596821, 0.005835224436425922, 0.007117635603196905, 0.009219441833998317, 0.008667261924007125, 0.006620082914951209, 0.006725658256161563, 0.007048395111077593, 0.009807754861203838, 0.006979959722577554, 0.007612773358800719, 0.006699697844786428, 0.008927195143870704, 0.009001352204206453, 0.006868189328638678, 0.006287586444771083, 0.005976617372269917, 0.007523386183492156, 0.007540073096341286, 0.008178306025099566, 0.007392601045038195, 0.005989636453951404, 0.0069656677154090775, 0.007692476865167692, 0.008839115373405072, 0.0066096871470014045, 0.006051266720514978, 0.007510484741231428, 0.006468628650691967, 0.008997133843821566, 0.007091105605050539, 0.006978069739861409, 0.006397553289710287, 0.0063910197301673575, 0.0073174569281525695, 0.006288037904211389, 0.0067849457229257155, 0.006049494914001016, 0.006461062690895201, 0.007957239221556485, 0.006741012822788855, 0.006654785293279676, 0.007001488375867896, 0.007449320893045436, 0.008346061395992041, 0.006707273323615231, 0.006320594472203786, 0.006792003644174724, 0.0066553387973114116, 0.007857285360498704, 0.007668383064872233, 0.007476240933391072, 0.006712512027757092, 0.006558765676184169, 0.008150126061820695, 0.0066711266625496335, 0.006943824172679121, 0.0069442655111361384, 0.007595752806991744, 0.007188349157454997, 0.006938922162655432, 0.007179886783185985, 0.0072626336866672345, 0.0073506279226075825, 0.0073144816145109615, 0.006685698010493656, 0.006582886537761213, 0.007072983377758776, 0.006283306430282448, 0.007865368292261506, 0.007235910472968399, 0.006812099663540832, 0.0070706840933989585, 0.006731582070680087, 0.00899652430466894, 0.007761627707808892, 0.00773477672328862, 0.0067844569005364695, 0.006694148746795523, 0.00844262320353014, 0.008163046434702826, 0.006883794355388318, 0.006728763348291852, 0.007558330156698353, 0.006977196835450918, 0.007858066968797693, 0.0072034336903549654, 0.007717883148796079, 0.006061343664734455, 0.008601543019545752, 0.007082528481263614, 0.0066558995398429644, 0.007288961491011314, 0.00829056240667179, 0.007802631186544282, 0.0075600041693374455, 0.006731572716024786, 0.006699391577615667, 0.007906832193760262, 0.008396308402314373, 0.00783886548628047, 0.006531806586242449, 0.007371469029063046, 0.0071365983937920026, 0.008348498312756707, 0.00826909237708374, 0.007032637128831787, 0.007372248791804411, 0.0073318599364706185, 0.005507784298926339, 0.005876256091700915, 0.007137037565558904, 0.007013423918433507, 0.008072497620863367, 0.007203022957580516, 0.004977451338000584, 0.006163760824387628, 0.006915677029031105, 0.007368005399611698, 0.005436400959376836, 0.005890498190477393, 0.005048539795044739, 0.006353970138759007, 0.006910673644549731, 0.013284437792567919, 0.006322443374125548, 0.0055570327714634625, 0.005084636879020193, 0.0064080319881656194, 0.010550685719904538, 0.007386710653358952, 0.00739074741488792, 0.005742522168158005, 0.005202776542601124, 0.0057122229743655495, 0.006158389621844955, 0.005294288604318729, 0.005877964359460478, 0.007876741450713987, 0.014509193897417927, 0.0072957831295326825, 0.00536813497886014, 0.0053792457060517105, 0.006429845015333747, 0.00912249445283647, 0.00580211761650988, 0.0049039078230787925, 0.006122930618599374, 0.004844274791497367, 0.013006125148338946, 0.0073756585146115316, 0.005196182031519903, 0.006216752437289944, 0.005336083845493771, 0.013575341302547901, 0.007035830780676805, 0.0059860986427409555, 0.006652244686413664, 0.0043693186605977515, 0.006542835382975458, 0.005430920837272537, 0.0050228392644958335, 0.0058834417682841434, 0.005671221318878257, 0.006243796029083371, 0.006482091986102091, 0.00657104772358161, 0.007172231597071563, 0.005417126436891292, 0.0072439090067749875, 0.009994679032427754, 0.006277200638455275, 0.006756776282813765, 0.005209307463986057, 0.0076649891428114845, 0.0084153360778644, 0.00574563328571522, 0.00489383778715042, 0.004995809900998689, 0.01948815352427737, 0.011905944985613462, 0.008381046590880376, 0.008473783087597074, 0.007077551641409849, 0.007920617018009347, 0.00675361957563013, 0.007013537850982752, 0.00805131609224495, 0.006141332490737335, 0.008177760984611678, 0.005296147470344149, 0.0068129169228047124, 0.006941849607551576, 0.004470414603520794, 0.0071248976000427, 0.005394350485052175, 0.012225533939666503, 0.008647701422953865, 0.007316528593983032, 0.009975741423121976, 0.005552318481613024, 0.00959176770532931, 0.005761092266164729, 0.007495146148852623, 0.13301099604004096, 0.0067804694830024955, 0.005753681752904163, 0.008876082966755815, 0.010092287632304192, 0.006584213986346788, 0.007048497495527529, 0.007453953154723895, 0.004768939739854468, 0.008780555863928336, 0.013249343657569398, 0.02279038710299341, 0.0070348220088100695, 0.014290516620572723, 0.008617331820406979, 0.04865968558987765, 0.009220295048135096, 0.024807391543542858, 0.009280819952101673, 0.006768479586831113, 0.010348286577810338, 0.013631708179954507, 0.012841005065991626, 0.007594962621869793, 0.007125020518104804, 0.7596591349395453, 0.008314529420989309, 0.009691051469202254, 0.010091481132177478, 0.006860737104459401, 0.00650964878601826, 0.005686445385594895, 0.00518772214149222, 0.00735335596837297, 0.008599052977389719, 0.009782377969948393, 0.0053849368400502974, 0.005148175743544655, 0.005475671915847575, 0.00684101045381378, 0.011512614854827527, 0.006222392048645972, 0.004607507250544972, 0.005513575061705609, 0.0066264361947558975, 0.00819796885431444, 0.007411464461703122, 0.005899807687027441, 0.004898131643773959, 0.005705311989176175, 0.01442594198191438, 0.007962095826119002, 0.0057393203396921754, 0.005970349302156549, 0.005745377092142429, 0.006130262491926536, 0.004799763800284742, 0.005843859686910796, 0.0060587135563775095, 0.006086581829200248, 0.007239966740607308, 0.014276356149352675, 0.005815275594609435, 0.004830726988491379, 0.005254245201951262, 0.007208068520631293, 0.004876343263490938, 0.005538850365620882, 0.005373460573205128, 0.0049166647492963145, 0.02043651923884236, 0.007590807007984639, 0.007508353727765998, 0.00431533493373507, 0.005612526887286369, 0.013699565155888965, 0.00882979643968286, 0.005809681271806599, 0.004071682614899814, 0.006521177471067422, 0.021914583974693688, 0.004918743526334493, 0.004409366814636904, 0.004784263985267191, 0.0063166247382576745, 0.011177990378504912, 0.008403679285153362, 0.006565680910267917, 0.007206713970066638, 0.0053399080203328335, 0.012053091659298602, 0.006605070578830404, 0.005464461224409742, 0.005685473303393974, 0.008348930685338986, 0.016431214630994662, 0.012648862566886236, 0.016552336344656617, 0.006654269526540785, 0.004865139979672688, 0.008973364769666078, 0.01642339018580938, 0.004882162577716877, 0.006256805050738823, 0.0072748844153190115, 0.01261955533240709, 0.009333074784068923, 0.00722947526295616, 0.005277765726907181, 0.007486831156858058, 0.00933508143753303, 0.01593236827370736, 0.018758031311950137, 0.005471423950304591, 0.006783312875037168, 0.009665578788515252, 0.008238806958115713, 0.00725506240329199, 0.004536852107838599, 0.010662168646941611, 0.01939611825562641, 0.006283955352151194, 0.013552899891536118, 0.00649031027402549, 0.007442628000740192, 0.012383357384549702, 0.055748393706107874, 0.009246270883390403, 0.030392944024092556, 0.017456590443084938, 0.01571410060459717, 0.0066313484582364605, 0.004564155769058759, 0.005576586965852968, 0.00507874404628442, 0.017280386035356306, 0.010750384839615665, 0.016569636334647034, 0.008907630710074443, 0.004052561719443669, 0.03134779119678107, 0.016914825736222817, 0.007606420822120047, 0.016636107326530387, 0.0057394937554871425, 0.017243176287301164, 0.011686784971249977, 0.007899987685775165, 0.008647324805668467, 0.007613010248293556, 0.013561993351515847, 0.0069634254369232325, 0.025623492124322713, 0.021914873773660245, 0.00529357719310116, 0.013499578537348804, 0.005432267244358813, 0.005319525239000988, 0.007320852382474683, 0.008216791162836223, 0.01135720815284276, 0.006800544238812343, 0.006179473886290355, 0.0065817472597003756, 0.006439872619123516, 0.00999423437892056, 0.011186008684293897, 0.006437423942489792, 0.005228715094895471, 0.005118614730677832, 0.015048065207740206, 0.008700740291164518, 0.006315533353660553, 0.005714576150798199, 0.005530522635279187, 0.009325409073948987, 0.0073580964266080524, 0.004671603303458508, 0.005411785152262478, 0.00474328096799308, 0.008726096650487417, 0.005316857714446223, 0.00439215613788738, 0.005369207810465979, 0.006751529806689145, 0.03709172241214461, 0.007166197087432854, 0.006934795198681291, 0.0056100284388795095, 0.007023932464470545, 0.010147379512729766, 0.016491211266959484, 0.006473628381420207, 0.005966590973698188, 0.0043440974483603245, 0.03588137800627523, 0.010578216801526518, 0.011949353685871163, 0.005963894899353876, 0.007595845902954576, 0.010086404320435853, 0.012328321773709815, 0.010683876830908367, 0.010964641430486803, 0.010474475677431061, 0.013045662806011328, 0.006485369895136668, 0.009787422061099818, 0.007114108598824748, 0.005362094773120847, 0.02641495444578373, 0.017176082669046387, 0.009386308224239514, 0.006342006133999606, 0.007434180108421125, 0.013447754913857668, 0.016490233866105312, 0.009167032913777646, 0.006346991204426077, 0.0038476642695850823, 0.019693155214546468, 0.023164282647784532, 0.012371570171558896, 0.015513827320825714, 0.005193592789981904, 0.018481835732024485, 0.016033058755080935, 0.023712044429588602, 0.025162381154306843, 0.01330301589823566, 0.043081398253580795, 0.006293256384673048, 0.01746909106372808, 0.006816370538774792, 0.0048630294434141436, 0.03669479796863626, 0.009445917657323073, 0.02020137770138158, 0.008627186184543988, 0.005651487826530191, 0.018387480771349014, 0.005661713865469042, 0.023147681910266022, 0.052678521354000696, 0.013362974187853552, 0.04030630674204981, 0.012565470781434612, 0.007063714593000395, 0.016659037833406554, 0.009857752299199484, 0.016025858366694734, 0.024907596053364456, 0.018700406136628777, 0.011462937940822005, 0.01790358095858239, 0.023669641511622472, 0.02285344679376103, 0.017136506600532172, 0.013195866126084616, 0.008643681900268332, 0.034346301964341235, 0.015087728934651592, 0.01916867243400441, 0.02769652793980037, 0.004478109397402975, 0.09009416765026683, 0.02174295285026207, 0.01773230660749392, 0.009302168032644822, 0.010533491857949438, 0.15756070112408194, 0.020794566759538644, 0.018001942900988017, 0.012075978068689182, 0.016995185227749728, 0.11711574380314392, 0.009531074526352226, 0.02218919212411234, 0.011342125585328475, 0.01771903454715494]

sorted_indexes = np.argsort(data_store) # Returns a list of original indexes if the array were to be sorted
for i in range (0,8):
    min_index = sorted_indexes[i]
    config = hyperparam_list_test[min_index]
    print(i+1,") Params: HN = ", config[0], " Epochs = ", config[1]," LR = ", config[2]," Batch Size = ", config[3])
    print("MSE: ", data_store[min_index], "\n")
    
print (np.linspace(0.002, 0.040,int(0.038/0.002)+1))

"""
#LR = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
#LR = np.linspace(0.002, 0.0200,int(0.018/0.002)+1) 
# HNx = [
#     (2, 2), (2, 4), (2, 8), (2, 12), (2, 16), (2, 20), 
#     (4, 2), (4, 4), (4, 8), (4, 12), (4, 16), (4, 20), 
#     (8, 2), (8, 4), (8, 8), (8, 12), (8, 16), (8, 20),
#     (12, 2), (12, 4), (12, 8), (12, 12), (12, 16), (12, 20),
#     (16, 2), (16, 4), (16, 8), (16, 12), (16, 16), (16, 20),
#     (20, 2), (20, 4), (20, 8), (20, 12), (20, 16), (20, 20)
#     ]
# HN = [4,6,8,10]