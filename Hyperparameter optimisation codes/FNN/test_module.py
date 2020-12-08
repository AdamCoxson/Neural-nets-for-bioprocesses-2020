import torch
import numpy as np
import pandas as pd

def test(test_inputs, test_labels, net):
    seq_length = 11 # Number of testable datapoints in each experimental trial (total length - 1)
    n_testing_sets = 5 # Number of expt. trials data sets used for testing data
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


