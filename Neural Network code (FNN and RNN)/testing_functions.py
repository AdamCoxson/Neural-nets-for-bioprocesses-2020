# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 16:35:40 2020
Author: Adam Coxson, MPhys Undergrad, The University of Manchester
Project: Evaluating hyperparameter optimisation techinques for an FNN and RNN
Module: testing_function.py
Dependancies: none

Note this is formatted for the neural networks used in my project so there is 
a fair bit of hard coding in this module. Either change them to work with new
data sets or generalise these functions.

Based on work by Mostafizor Rahman, see README.txt
"""
import numpy as np
import torch
from torch.autograd import Variable

def FNN_test(test_inputs, test_labels, net):
    """
    ARGS: test_inputs: testing data formatted for use with Torch package
          test_labels: testing data formatted for use with Torch package
          net: the trained FNN network to be tested
    OUTPUTS: avg_mse: the MSE value averaged over all testing data points
            testing_set_mse_vals: the MSE values averaged over each testing set of data
            predictions_offline: offline prediction data for each of testing sets
    """
    seq_length = 11    # Number of testable datapoints in each experimental trial (total length - 1)
    n_testing_sets = 4 # Number of expt. trials data sets used for testing data
    squared_error_X = []
    predictionNumpy = []
    experimental = []
    testing_set_mse_vals = [0]*n_testing_sets
    net.eval()
    test_X = torch.Tensor(test_inputs).view(-1, 4)
    test_y = torch.Tensor(test_labels)

    with torch.no_grad(): # Obtaining prediction outputs
        for i in range(0, len(test_X)):
            net_out = net(test_X[i].view(-1, 4))
            predictionNumpy.append(net_out[0].numpy())              

    for data in test_y:
        experimental.append(data.numpy())
    # Finding the average Mean squared Errors
    for i in range(0, len(experimental)):
            X_error = experimental[i][0] - predictionNumpy[i][0]
            squared_error_X.append(X_error**2)
            
    for i in range(0,n_testing_sets):
        testing_set_mse_vals[i] = sum(squared_error_X[0+(i*seq_length):seq_length+(i*seq_length)])/seq_length
    
    #avg_mse = sum(squared_error_X[:])/len(squared_error_X)
    avg_mse = np.mean(np.array(testing_set_mse_vals))
    
    # Forming prediction data for biomass conc. Gas areation rate, CO2 conc. and Light intensity.
    bc_init, ga, co, li = [], [], [], [] # biomass concentration is the only non-constant
    for i in range(0, n_testing_sets):
        bc_init.append(test_inputs[0+(i*seq_length)][0])
        ga.append(test_inputs[0+(i*seq_length)][1])
        co.append(test_inputs[0+(i*seq_length)][2])
        li.append(test_inputs[0+(i*seq_length)][3])
        
    predictions_offline = []
    for i in range(0, n_testing_sets):
        bc = bc_init[i]
        predictions_offline.append([float(bc), float(ga[i]), float(co[i]), float(li[i])])
        for j in range(0, seq_length):
            net_out = net(torch.Tensor([bc, ga[i], co[i], li[i]])) # Getting predictions outputs
            bc = bc + net_out[0]
            predictions_offline.append([float(bc), float(ga[i]), float(co[i]), float(li[i])])

    return avg_mse, testing_set_mse_vals, predictions_offline


def RNN_test(test_inputs, test_labels, net):
    """
    ARGS: test_inputs: testing data formatted for use with Torch package
          test_labels: testing data formatted for use with Torch package
          net: the trained FNN network to be tested
    OUTPUTS: avg_mse: the MSE value averaged over all testing data points
            testing_set_mse_vals: the MSE values averaged over each testing set of data
            predictions_offline: offline prediction data for each of testing sets
    """
    seq_length = 10 # Number of testable datapoints in each experimental trial (total length - 1)
    n_testing_sets = 4 # Number of expt. trials data sets used for testing data
    testing_set_mse_vals = [0]*n_testing_sets
    squared_error_X = []
    net.eval()
    test_X = Variable(torch.Tensor(test_inputs)) 
    test_y = Variable(torch.Tensor(test_labels))
    hidden = net.init_hidden(test_X)
    with torch.no_grad():
        net_out, _ = net(test_X, hidden)  # Hidden state not required for manual feeding


    for index1, element in enumerate(test_y):
        for index2, row in enumerate(element):
            X_error = row[0] - net_out[index1][index2][0]
            squared_error_X.append(float(X_error)**2)

    
    for i in range(0,n_testing_sets):
        testing_set_mse_vals[i] = sum(squared_error_X[0+(i*seq_length):seq_length+(i*seq_length)])/seq_length
    avg_mse = np.mean(np.array(testing_set_mse_vals))
    
    prediction = []
    bc_init, ga, co, li = [], [], [], [] # biomass concentration is the only non-constant variable
    for i in range(0, n_testing_sets):
        bc_init.append(test_inputs[i][0][0])
        ga.append(test_inputs[i][0][1])
        co.append(test_inputs[i][0][2])
        li.append(test_inputs[i][0][3])
        
    net.sequence_length = 1              # Feed 1 input at a time to the network (offline prediction) 
    for i in range(0,n_testing_sets):
        hidden = net.init_hidden(Variable(torch.Tensor([[[]]])))            # Initialise hidden state with a batch size of 1
        bc = bc_init[i]
        prediction.append([float(bc), float(ga[i]), float(co[i]), float(li[i])])
        for j in range(0, seq_length):
            # Feed inputs with a batch size of 1, sequence length of 1 and feature vector length of 5 to the network
            net_out, hidden = net(Variable(torch.Tensor([[[bc, ga[i], co[i], li[i]]]])), hidden)
            bc = bc + net_out[0][0][0]
            prediction.append([float(bc), float(ga[i]), float(co[i]), float(li[i])])

    return avg_mse, testing_set_mse_vals, prediction

def FNN_test_mse_only(test_inputs, test_labels, net):
    """
    ARGS: test_inputs: testing data formatted for use with Torch package
          test_labels: testing data formatted for use with Torch package
          net: the trained FNN network to be tested
    OUTPUTS: avg_mse: the MSE value averaged over all testing data points
    """
    seq_length = 11 # Number of testable datapoints in each experimental trial (total length - 1)
    n_testing_sets = 4 # Number of expt. trials data sets used for testing data
    squared_error_X = []
    predictionNumpy = []
    experimental = []
    testing_set_mse_vals = [0]*n_testing_sets
    net.eval()
    test_X = torch.Tensor(test_inputs).view(-1, 4)
    test_y = torch.Tensor(test_labels)

    with torch.no_grad(): # Obtaining prediction outputs
        for i in range(0, len(test_X)):
            net_out = net(test_X[i].view(-1, 4))
            predictionNumpy.append(net_out[0].numpy())              

    for data in test_y:
        experimental.append(data.numpy())
    # Finding the average Mean squared Errors
    for i in range(0, len(experimental)):
            X_error = experimental[i][0] - predictionNumpy[i][0]
            squared_error_X.append(X_error**2)
            
    for i in range(0,n_testing_sets):
        testing_set_mse_vals[i] = sum(squared_error_X[0+(i*seq_length):seq_length+(i*seq_length)])/seq_length
    avg_mse = np.mean(np.array(testing_set_mse_vals))
    
    return avg_mse

def RNN_test_mse_only(test_inputs, test_labels, net):
    """
    ARGS: test_inputs: testing data formatted for use with Torch package
          test_labels: testing data formatted for use with Torch package
          net: the trained FNN network to be tested
    OUTPUTS: avg_mse: the MSE value averaged over all testing data points
    """
    seq_length = 10 # Number of testable datapoints in each experimental trial (total length - 1) (HARDCODED)
    n_testing_sets = 4 # Number of expt. trials data sets used for testing data (HARDCODED)
    testing_set_mse_vals = [0]*n_testing_sets
    squared_error_X = []
    net.eval()
    test_X = Variable(torch.Tensor(test_inputs)) 
    test_y = Variable(torch.Tensor(test_labels))
    hidden = net.init_hidden(test_X)
    with torch.no_grad():
        net_out, _ = net(test_X, hidden)  # Hidden state not required for manual feeding

    for index1, element in enumerate(test_y):
        for index2, row in enumerate(element):
            X_error = row[0] - net_out[index1][index2][0]
            squared_error_X.append(float(X_error)**2)

    for i in range(0,n_testing_sets):
        testing_set_mse_vals[i] = sum(squared_error_X[0+(i*seq_length):seq_length+(i*seq_length)])/seq_length
    avg_mse = np.mean(np.array(testing_set_mse_vals))
    
    return avg_mse


