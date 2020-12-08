# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 21:19:29 2020
Author: Adam Coxson, MPhys Undergrad, The University of Manchester
Project: Evaluating hyperparameter optimisation techinques for Bioprocesses
"Finding the Optimal Optimiser"
Module: HPO_Bayesian_GP  (Gaussian Process Bayesian Optimisation)

An example I pulled from this link:
https://machinelearningmastery.com/what-is-bayesian-optimization/

"""

# example of bayesian optimization with scikit-optimize
import copy
import csv
import numpy as np
from cpuinfo import get_cpu_info
from skopt.space import Integer
from skopt.space import Real
from skopt.utils import use_named_args
from skopt import forest_minimize
from rnn import RNN
from train import train
from test_module import test
from datetime import datetime

def write_to_csv(filename, mse_values, x_iters, hyperparameter_names,time):
    
    for i in range(0,len(mse_values)):
        x_iters[i].insert(0,mse_values[i])
    hyperparameter_names.insert(0,"MSE")
    
    writer = csv.writer(open(filename,'w'),lineterminator ='\n')
    writer.writerow(["Time:", time, "Size:", len(mse_values),"CPU:", get_cpu_info()['brand_raw']])
    writer.writerow(hyperparameter_names)
    writer.writerows(x_iters)
    print("Written to",filename,"successfully.")

from data_preprocessing import data_preprocess
training_data, testing_data = data_preprocess()
training_inputs = training_data[:, 0:4]
training_labels = training_data[:, 4:]
test_inputs = testing_data[:, 0:4]
test_labels = testing_data[:, 4:]

training_inputs = np.split(training_inputs, 820)
training_labels = np.split(training_labels, 820)
test_inputs = np.split(test_inputs, 5)
test_labels = np.split(test_labels, 5)

bounds = [Integer(1, 20,  name='HN'),
          Integer(40, 400,  name='EPOCHS'), 
          Integer(1, 300,  name='BS'),
          Real(0.0001,0.1,  name='LR')]
@use_named_args(bounds)


def neural_net(HN, EPOCHS, BS, LR):
    HL = 1 # one layer RNN
    net = RNN(1, 4, 10, HN, HL)
    init_state = copy.deepcopy(net.state_dict())
    net.load_state_dict(init_state)
    train(net, training_inputs, training_labels, EPOCHS, LR, BS)
    avg_mse = float(test(test_inputs, test_labels, net))
    print(" MSE: ",round(avg_mse, 7),"HN:",HN,"  Epochs:", EPOCHS, "  Batch Size:", BS,"  Learn rate:", round(LR,6))
    return avg_mse

print("\nScript Started\n")
random_vals = 50
num_evaluations = random_vals + 200
# random_vals = 50
# num_evaluations = random_vals + 350
# random_vals = 2
# num_evaluations = random_vals + 3

init_time = datetime.now()
# Runs random forest minimisation using the extra trees base estimator by default.
result = forest_minimize(neural_net,            # the function to minimize
                     dimensions=bounds,    # the bounds on each dimension of x
                     acq_func="EI",        # the acquisition function
                     n_calls = num_evaluations,           # the number of evaluations of f
                     n_random_starts = random_vals,  # the number of random initialization points                       
                     random_state=1234)     # the random seed
# summarizing finding:
fin_time = datetime.now()
print("\nRandom Forest execution time: ", (fin_time-init_time), "(hrs:mins:secs)")
print('Best MSE: %.7f' % (result.fun))
print('Best Parameters HN: %d, Epochs: %d, Batch Size: %d, Learn rate: %.4f' % (result.x[0], result.x[1], result.x[2], result.x[3]))
#print('Best Parameters: Epochs = %d, Batch Size = %d, Learn rate = %.4f' % (result.x[0], result.x[1], 0.006))

# Saving data to file
hyperparams = ["HN","Epochs", "Batch Size", "Learning rate"]
#filename = 'Bayesian_GP n_rand:'+ str(random_vals) + ', n_evals:' + str(num_evaluations) + ', n_hyperparams:' + str(len(hyperparams))
filename = 'RandF_RNN_250_4'+'.csv'

# for i in range(0,num_evaluations): # If H1 and H2 are being tested
#     result.x_iters[i][0] = result.x_iters[i][0]*2
#     result.x_iters[i][1] = result.x_iters[i][1]*2
    
write_to_csv(filename,result.func_vals, result.x_iters, hyperparams, fin_time-init_time)




    
    
    
    
    
    