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
import pandas as pd
import numpy as np
import copy
from skopt.space import Integer
from skopt.space import Real
from skopt.utils import use_named_args
from skopt import gp_minimize
from ann2 import Net
from train3 import train
from test2 import test
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
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

from data_preprocessing import data_preprocess
training_data, testing_data = data_preprocess()
training_inputs = training_data[:, 0:5]
training_labels = training_data[:, 5:]
test_inputs = testing_data[:, 0:5]
test_labels = testing_data[:, 5:]


#plot_data = pd.DataFrame(index=None, columns="MSE EPOCHS BS LR".split())

# bounds = [Integer(10, 400,  name='EPOCHS'), 
#           Integer(10, 500,  name='BS'), 
#           Real(0.0001,0.3,  name='LR')]
bounds = [Integer(10, 500,  name='EPOCHS'), 
          Integer(10, 400,  name='BS')]
# define the function used to evaluate a given configuration
@use_named_args(bounds)

#def neural_net(EPOCHS, BS, LR):
def neural_net(EPOCHS, BS):
    #EPOCHS = 300
    #BS = 50
    h1 = 4
    h2 = 8
    LR = 0.006
        
    net = Net(h1, h2)
    init_state = copy.deepcopy(net.state_dict())
    net.load_state_dict(init_state)
    train(net, training_inputs, training_labels, EPOCHS, LR, BS)
    avg_mse = test(test_inputs, test_labels, net)
    
    print("MSE = ",round(avg_mse[0],7),", Epochs = ", EPOCHS, ", Batch Size = ", BS,", Learning rate = ", round(LR,7))
    return avg_mse[0]

print("\nScript Started\n")
random_vals = 30
num_evaluations = random_vals + 200
# random_vals = 2
# num_evaluations = random_vals + 3

init_time = datetime.now()
result = gp_minimize(neural_net,            # the function to minimize
                     dimensions=bounds,    # the bounds on each dimension of x
                     acq_func="EI",        # the acquisition function
                     n_calls = num_evaluations,           # the number of evaluations of f
                     n_random_starts = random_vals,  # the number of random initialization points                       
                     random_state=12)     # the random seed
# summarizing finding:
fin_time = datetime.now()
print("\nBayesian GP Optimisation execution time: ", (fin_time-init_time), "(hrs:mins:secs)")
print('Best MSE: %.7f' % (result.fun))
#print('Best Parameters: Epochs = %d, Batch Size = %d, Learn rate = %.4f' % (result.x[0], result.x[1], result.x[2]))
print('Best Parameters: Epochs = %d, Batch Size = %d, Learn rate = %.4f' % (result.x[0], result.x[1], 0.006))

# Saving data to file
hyperparams = ["Epochs", "Batch Size"]
#filename = 'Bayesian_GP n_rand:'+ str(random_vals) + ', n_evals:' + str(num_evaluations) + ', n_hyperparams:' + str(len(hyperparams))
filename = 'Bayesian_GP_data_4'+'.csv'
write_to_csv(filename,result.func_vals, result.x_iters, hyperparams, fin_time-init_time)

"""
I need to create a 2D grid of MSE values before I can overplot the MSE outputs
from each Bayesian Optimisation iteration
"""

# cols = ['#f1eef6','#d0d1e6','#a6bddb','#74a9cf','#3690c0','#0570b0','#034e7b']
# levs = [0.17,0.175,0.18,0.19,0.2,0.25,0.3,1]

# xs = np.linspace(10,500,491)
# ys = np.linspace(10,400,391)
# xv, yv = np.meshgrid(xs, ys)
# xv = np.concatenate(xv)
# yv = np.concatenate(yv)
# x_list = y_list = w_list = []

# data = np.array(result.x_iters).transpose()
# x_list = data[0][:]
# y_list = data[1][:]
# w_list = np.squeeze(result.func_vals)
# #fig, axe = plt.subplots(figsize=(8,6))
# fig, axe = plt.subplots()
# axe.set_xlabel('Epochs')
# axe.set_ylabel('Batch size')
# plot1 = axe.contourf(x_list, y_list, w_list, levs, colors=cols)
# axe.scatter(xv, yv, color='k', marker='x')
# cb = plt.colorbar(plot1, ax=axe, extend='both')
# cb.set_label('MSE, %')
# #make_circ(axe,80,60,5)
# fig.tight_layout()
# #plt.savefig('test.png')


    
    
    
    
    
    