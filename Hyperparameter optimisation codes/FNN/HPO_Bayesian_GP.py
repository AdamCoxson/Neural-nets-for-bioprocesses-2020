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
from skopt.space import Integer
from skopt.space import Real
from skopt.utils import use_named_args
from skopt import gp_minimize
from ann import Net
from train_v2 import train
from test import test
from datetime import datetime
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

from data_preprocessing_B import data_preprocess
training_data, testing_data = data_preprocess()
training_inputs = training_data[:, 0:4]
training_labels = training_data[:, 4:]
test_inputs = testing_data[:, 0:4]
test_labels = testing_data[:, 4:]

bounds = [Integer(1, 5,  name='H1'),
          Integer(1, 2,  name='H2'),
          Integer(1, 400,  name='EPOCHS'), 
          Integer(1, 300,  name='BS'),
          Real(0.0001,0.2,  name='LR')]
@use_named_args(bounds)

# define the function used to evaluate a given configuration
#def neural_net(EPOCHS, BS, LR):
#def neural_net(EPOCHS, BS):
    #EPOCHS = 300
    #BS = 50
    #h1 = 4
    #h2 = 8
    #LR = 0.006
def neural_net(EPOCHS, BS, LR, H1, H2):
    h1 = 2*H1
    h2 = 2*H2
        
    net = Net(h1, h2)
    init_state = copy.deepcopy(net.state_dict())
    net.load_state_dict(init_state)
    train(net, training_inputs, training_labels, EPOCHS, LR, BS)
    avg_mse = test(test_inputs, test_labels, net)
    print(" MSE: ",round(avg_mse[0],7),"HN: (",h1,",",h2,") Epochs: ", EPOCHS, "Batch Size: ", BS," Learn rate: ", round(LR,6))
    return avg_mse[0]

print("\nScript Started\n")
random_vals = 50
num_evaluations = random_vals + 350
# random_vals = 2
# num_evaluations = random_vals + 3

init_time = datetime.now()
result = gp_minimize(neural_net,            # the function to minimize
                     dimensions=bounds,    # the bounds on each dimension of x
                     acq_func="EI",        # the acquisition function
                     n_calls = num_evaluations,           # the number of evaluations of f
                     n_random_starts = random_vals,  # the number of random initialization points                       
                     random_state=1234)     # the random seed
# summarizing finding:
fin_time = datetime.now()
print("\nBayesian GP Optimisation execution time: ", (fin_time-init_time), "(hrs:mins:secs)")
print('Best MSE: %.7f' % (result.fun))
print('Best Parameters HN: (%d, %d), Epochs: %d, Batch Size: %d, Learn rate: %.4f' % (result.x[0]*2, result.x[1]*2, result.x[2], result.x[3], result.x[4]))
#print('Best Parameters: Epochs = %d, Batch Size = %d, Learn rate = %.4f' % (result.x[0], result.x[1], 0.006))

# Saving data to file
hyperparams = ["H1","H2","Epochs", "Batch Size", "Learning rate"]
#filename = 'Bayesian_GP n_rand:'+ str(random_vals) + ', n_evals:' + str(num_evaluations) + ', n_hyperparams:' + str(len(hyperparams))
filename = 'Bayesian_GP_400evals_4'+'.csv'

for i in range(0,num_evaluations): # If H1 and H2 are being tested
    result.x_iters[i][0] = result.x_iters[i][0]*2
    result.x_iters[i][1] = result.x_iters[i][1]*2
    
write_to_csv(filename,result.func_vals, result.x_iters, hyperparams, fin_time-init_time)




    
    
    
    
    
    