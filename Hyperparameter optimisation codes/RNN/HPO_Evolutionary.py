# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 00:27:47 2020

@author: adamc
"""

import numpy as np 
import numpy.random as rnd
import matplotlib.pyplot as plt 
import copy 
import time
import pandas as pd
from ann_B import Net
from train_Bv2 import train
from test_B import test
from datetime import datetime
import csv

from data_preprocessing_B import data_preprocess
training_data, testing_data = data_preprocess()
training_inputs = training_data[:, 0:4]
training_labels = training_data[:, 4:]
test_inputs = testing_data[:, 0:4]
test_labels = testing_data[:, 4:]

i = 1
def write_to_csv(filename, mse_values, x_iters, hyperparameter_names,time):
    
    for i in range(0,len(mse_values)):
        x_iters[i].insert(0,mse_values[i])
    hyperparameter_names.insert(0,"MSE")
    
    writer = csv.writer(open(filename,'w'),lineterminator ='\n')
    writer.writerow(["Time:", time])
    writer.writerow(hyperparameter_names)
    writer.writerows(x_iters)
    print("Written to",filename,"successfully.")

'''
Evolutionary Algorithm - Tom Savage 

--IMPLEMENTATION NOTES--

i_pos:  'individual positions' 

i_pos is the list of population coordinates
the last column (axis 1) is that rows function values
(this operation is performed in the func_eval function)

crossover and mutation are done with this column in place
then it is updated at the end 

tournament selection used, the tournament size can be changed
'''


def LHS(bounds,p):
    '''
    INPUTS: 
    bounds      : bounds of function to be optimized 
                  in form [[xl,xu],[xl,xu],[xl,xu]]
    p           : population size 
    
    OUTPUTS: 
    sample      : LHS sample, rows are solutions, cols are vars    
                e.g. [[x11,x12],[x21,x22]]
    '''
    d = len(bounds)                     # dimensions
    sample = np.zeros((p,len(bounds)))  # memory allocation
    for i in range(0,d):
        # produce a grid of numbers in each col (variable)
        #sample[:,i] = np.round(np.linspace(bounds[i,0],bounds[i,1],p))
        sample[:,i] = (np.linspace(bounds[i,0],bounds[i,1],p))
        # shuffle the grid
        rnd.shuffle(sample[:,i])
    # results in no 'repeat' samples in any dimension
    return sample 

def rand_sample(bounds,p):
    '''
    INPUTS: 
    bounds      : bounds of function to be optimized 
                  in form [[xl,xu],[xl,xu],[xl,xu]]
    p           : population size 
    
    OUTPUTS: 
    sample      : random sample, rows are solutions, cols are vars  
                e.g. [[x11,x12],[x21,x22]]  
    '''    
    d = len(bounds)
    sample = np.zeros((p,len(bounds)))
    for i in range(0,d):
        # randomly sampling in each dimension
        sample[:,i] = rnd.uniform(bounds[i,0],bounds[i,1],p)
    return sample 

def func_eval(f,i_pos):
    '''
    INPUTS: 
    f       : function to optimise
    i_pos   : matrix of population WITHOUT FUNCTION COLUMN
    
    OUTPUTS: 
    i_pos   : matrix of population with function column appended
             e.g. [[x11,x12,f1],[x21,x22,f2]] 
    '''      
    # allocating memory for function values
    i_val = np.zeros((len(i_pos),1)) 
    for i in range(len(i_pos)):
        # assigning each a function value
        i_val[i,:] = f(i_pos[i,:])
    # appending func column to individual positions
    i_pos = np.concatenate((i_pos,i_val),axis=1)
    return i_pos

def tournament_selection(cull_percen,p,d,i_pos):
    '''
    INPUTS: 
    cull_percen : percentage of individuals to lose
    p           : population size
    d           : dimensions 
    i_pos       : individual position matrix
    
    OUTPUTS: 
    i_pos       : matrix of selected individuals 
    '''      
    # assigning memory for selected population
    i_new     = np.zeros((int(p*(cull_percen)),d+1)) 
    new_count = 0
    # whilst selected population is non-full
    while new_count < len(i_new):
        rnd.shuffle(i_pos)
        t_size = 2 
        # create tournament from first 2 individuals
        t=i_pos[:t_size,:]
        # select best based on func value (last column)
        t_best=t[np.argmin(t[:,-1])]
        # append to latest population
        i_new[new_count,:]=t_best[:]
        new_count+=1
    i_pos=copy.deepcopy(i_new)
    return i_pos

def crossover(i_pos,d):
    '''
    INPUTS: 
    i_pos       : individual position matrix
    d           : dimensions
    
    OUTPUTS: 
    i_pos       : matrix of crossed over individuals 
    '''   
    rnd.shuffle(i_pos)
    p           = len(i_pos)
    cross_index = np.linspace(0,p-2,int(p/2))
    parents     = np.copy(i_pos)
    for i in cross_index:
        i = int(i)
        # choosing a random place to cross
        k = rnd.randint(0,d)
        # crossing first parent
        i_pos[i+1,k:] = i_pos[i,k:]
        # crossing second parent
        i_pos[i,k:]   = parents[i+1,k:]
    # appending parents and children into one matrix 
    i_pos = np.concatenate((i_pos,parents),axis=0)
    return i_pos 

def mutation(i_pos,d,mut_percen,bounds):
    '''
    INPUTS: 
    i_pos       : individual position matrix
    d           : dimensions
    mut_percen  : percentage chance of mutation 
    
    OUTPUTS: 
    i_pos       : matrix of mutated individuals 
    '''   
    # over each individual, over each dimension
    for i in range(len(i_pos)):
        for j in range(d):
            # get random num ~{0...1}
            prob = rnd.uniform()
            if prob < mut_percen:
                # assign random value between dimension bounds
                i_pos[i,j] = rnd.uniform(bounds[j,0],bounds[j,1])
            return i_pos

def evolution(f,bounds,p,it,cull_percen,mut_percen):
    '''
    INPUTS: 
    f           : function to be optimized
    bounds      : bounds of function to be optimized 
                  in form [[xl,xu],[xl,xu],[xl,xu]]
    p           : population size 
    it          : number of generations 
    cull_percen : percentage of particles to be culled after each generation
    mut_percen  : percentage chance of a mutation to occur 

    OUTPUTS: 
    returns the coordinates of the best individual
    '''
    d = len(bounds)
    #---ORIGINAL POPULATION SAMPLE---#
    i_pos = LHS(bounds,p)
    #---EVALUATING FITNESSES---------#
    i_pos     = func_eval(f,i_pos)
    i_best    = i_pos[np.argmin(i_pos[:,-1])]
    iteration = 0
    while iteration < it: 
    #---TOURNAMENT SELECTION---------#
        i_pos = tournament_selection(cull_percen,p,d,i_pos)
    #---COMPLETING WITH RANDOM CANDIDATES---#
        new_psize = p - len(i_pos) # get the number needed
        i_new     = rand_sample(bounds,new_psize) # random sample
        i_new     = func_eval(f,i_new) # evaluate function for new individuals
        i_pos     = np.concatenate((i_new,i_pos),axis=0) # append to population
    #---PRINTING BEST--------------#
        best_index = np.argmin(i_pos[:,-1])
        i_best     = i_pos[best_index]
        i_best_val = i_pos[best_index,-1]
        print(i_best_val,end='\r')
    #---CROSSOVER HERE-------------#
        i_pos = crossover(i_pos,d)
    #---MUTATION HERE--------------#
        i_pos = mutation(i_pos,d,mut_percen,bounds)
    #---UPDATING FUNCTION VALUES---#
        i_pos = i_pos[:,:-1] # remove meaningless function column 
        i_pos = func_eval(f,i_pos) # re-evaluate function values
        iteration += 1 # update iteration count
    return i_best

def neural_net(Hyperparams):
    EPOCHS = int(Hyperparams[0])
    LR = float(Hyperparams[1])
    BS = 60
    h1 = 4
    h2 = 8
    
    global i
    net = Net(h1, h2)
    init_state = copy.deepcopy(net.state_dict())
    net.load_state_dict(init_state)
    train(net, training_inputs, training_labels, EPOCHS, LR, BS)
    avg_mse = test(test_inputs, test_labels, net)
    
    print(i,") MSE = ",round(avg_mse[0],7),", Epochs = ", EPOCHS, ", Batch Size = ", BS,", Learning rate = ", round(LR,7))
    i=i+1
    return avg_mse[0]

""" ---------------------- MAIN ---------------------- """
p = 60            # population size
f = neural_net    # Objective function to be optimised
#---bounds for optimisation--–#
bounds = np.array([[10, 500],[0.0001,0.3]])
iterations = 10 # iterations 
#---percentage chance of a mutation---#
mutation_percent = 0.05 
#---percentage of populations to be 'killed---#
cull_percent = 0.95 


print(evolution(neural_net,bounds,p,iterations,cull_percent,mutation_percent))


