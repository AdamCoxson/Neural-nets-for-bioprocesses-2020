# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 16:54:01 2020
Author: Adam Coxson, MPhys Undergrad, The University of Manchester
Project: Evaluating hyperparameter optimisation techinques for an FNN and RNN
Module: ann_hyperparameter_evaluation 
Dependancies: neural_networks, network_train_and_test_module, training_function, testing_functions, data_preprocessing
             
This script is for evaluating sets of hyperparameters which have bee nfound previously to be potential optimal
by different hyperparameter optimisation techniques. This script allows for evaluating these sets multiple 
times to average the data and identifies any anomalous outliers which were flagged as optimal due to having
extremely high variance (such as MSE: 0.1 ± 0.08). Parallel processing is used to speed up the execution time
for a high number of samples. For effective evaluation of mean and standard deviation, use n_samples = 100. For
faster processing use n_samples = 20, which is not as accurate but it is large enough to get within 1 std. This
can be applied to both the FNN and RNN architectures used in this project. The Hyperparameters are defined by 
the user in MAIN and the program outputs the results to the console as well as produce prediction plots for 
each configuration.

"""
# Packages 
import numpy as np                
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import copy
from datetime import datetime
from joblib import Parallel, delayed
 # Modules 
from neural_networks import fnn_net, rnn_net             
from network_train_and_test_module import train_and_test, closest_to_average, print_results
from data_preprocessing import data_preprocess

      
def neural_network_evaluation(cfg, expt_data, test_scaler,  cores, n_samples, network_type):
    """
    ARGS: cfg: A single configuration of hyperparameters (Neuron config, Epochs, batch size, learning rate),
          expt_data: formatted and normalised training and testing data  
          test_scaler: Standard Scaler function fitted to the testing data
          cores: number of cores to be used in parallel processing,
          n_samples: number of samples to average over for a given set of hyperparameters,
          network_type: either 'fnn' or 'rnn'.
          
    OUTPUTS: avg_mse: average of all the different, n_sample test mse values, 
             std_mse: spread of all the different, n_sample test mse values, 
             train_mse: average of all the different, n_sample training mse values,  
             train_std: spread of all the different, n_sample training mse values, 
             mse_vals: list of all the testing mse value (each an average of the test sets), 
             execution time,
             testing_set: mse values from the individual testing sets from the mse_val closest to the avg val.
             prediction: network prediction data corresponding to the outputted testing set.
             (outputs all passed out as a tuple)
             
    This function takes in a set of hyperparameters and evaluates them for the FNN or RNN architectures developed in the 
    program. It repeats the network training and testing n_samples times over and averages this data, utilising parallel
    processing via the train_and_test() function. This is used to further evaluate optimal hyperparameters
    identifed from previous analyse to identify any outliers wrongly believed to be optimal, as many hyperparameter sets 
    have a very high variance after training and testing. For final values of mean and standard deviation for a given 
    hyperparameter set, it is recommended to use 40 - 100 samples. The outputted prediction data is taken from the 
    iteration of testing mse which was closest to the calculated avg (see closest_to_average func). The parallel 
    processing used is from the joblib package which automatically deals with multiple processes and data reorganisation 
    after processing.
    """
    start_time = datetime.now()
    if network_type == 'fnn': 
        H1     = int(cfg[0])
        H2     = int(cfg[1])
        EPOCHS = int(cfg[2])
        bs     = int(cfg[3])
        lr     = float(cfg[4])
    elif network_type == 'rnn':
        HL     = 1
        HN     = int(cfg[0])
        EPOCHS = int(cfg[1])
        bs     = int(cfg[2])
        lr     = float(cfg[3])
    else:
        print("Invalid neural network type, choose either fnn or rnn")
        exit(1)
    mse_vals          = np.zeros(n_samples)
    training_mse_vals = [0]*n_samples
    results           = [0]*n_samples
    networks          = [0]*n_samples
    
    for i in range(0,n_samples):
        if network_type =='fnn':
            net = fnn_net(H1, H2)
        else:
            net = rnn_net(1, 4, 10, HN, HL)     
        init_state = copy.deepcopy(net.state_dict())
        net.load_state_dict(init_state)
        networks[i] = net
    # Joblib parallelisation across the number of samples
    results[:] = Parallel(n_jobs=cores)(delayed(train_and_test)(EPOCHS, lr, bs, expt_data, network_type, net) for net in networks)
    for i in range (0,n_samples):
        mse_vals[i] = results[i][0]
        training_mse_vals[i] = np.array(float(results[i][1]))
        
    avg_mse   = np.mean(mse_vals)
    std_mse   = np.std(mse_vals)
    train_mse = np.mean(training_mse_vals) 
    train_std = np.std(training_mse_vals)
    avg_idx   = closest_to_average(mse_vals, np.mean(mse_vals))
    testing_set = np.sqrt(np.array(results[avg_idx][2]))
    predictions_inverse = test_scaler.inverse_transform(results[avg_idx][3])
    prediction = predictions_inverse.T[0][:]
    end_time = datetime.now()
    
    if network_type == 'fnn':
        print('\nParameters HN: (%d, %d), Epochs: %d, Batch Size: %d, Learn rate: %.6f'%(cfg[0],cfg[1],cfg[2],cfg[3],cfg[4]))
    if network_type == 'rnn':
        print('\nParameters HN: %d, Epochs: %d, Batch Size: %d, Learn rate: %.6f'%(cfg[0],cfg[1],cfg[2],cfg[3]))
    print("MSE values:\n",(mse_vals))
    print('Mean Squared Error: (%.4f \u00b1 %.4f), Mean Error: (%.4f \u00b1 %.6f)'%(avg_mse,std_mse, np.mean(np.sqrt(mse_vals)),np.std(np.sqrt(mse_vals))))
    print('Training Mean Squared Error: (%.4f \u00b1 %.4f)'%(train_mse, train_std))
    print("Training execution time: ", (end_time-start_time), "(hrs:mins:secs)")
    return (avg_mse, std_mse, train_mse, train_std, mse_vals, (end_time-start_time), testing_set, prediction)



##### MAIN ######
"""
2-layer FNN hyperparameter lists: H1, H2 Epochs, Batch size, learning rate.
"""
#hyperparam_list = [[8,4,186,20,0.0001],[8,4,141,456,0.0010], [8,4,245,55,0.00015],[8,4,510,200,0.00015],[6,2,586,400,0.000311],[4,4,800,100,0.0001]]
#hyperparam_list = [[8,4,245,55,0.00015]]
#hyperparam_list = [[8,4,245,55,0.00015],[8,4,510,200,0.00015],[8,4,186,20,0.0001],[4,4,800,100,0.00010]]
network_type = 'fnn'

"""
1-layer RNN hyperparameter lists: HN, Epochs, Batch size, learning rate.
"""
hyperparam_list = [[10,50,121,0.002913],[23,96,103,0.000762],[25,36,1,0.000106],[30,141,77,0.000443]]
network_type = 'rnn'
cores = 4          # Number of cores for multiprocessing.
n_samples = 100      # Number of evaluations per hyper-parameter set. For final results I recommend 100, or at least 20
biomass_predictions = [0]*len(hyperparam_list) # list initialisation
avg_mse             = [0]*len(hyperparam_list)
testing_set_data    = [0]*len(hyperparam_list)
results             = [0]*len(hyperparam_list)


init_time = datetime.now()
# This section obtains the training data and runs the neural networks through the hyperparameters.
if network_type == 'fnn': 
    timestamps = [0,12,24,36,48,60,72,84,96,108,120,132] # Hard coded time steps for the FNN testing data
    seq_length = 12 # Number of data points passed into network. 13th is trimmed off due to some expts only having 11 data points
elif network_type == 'rnn':
    timestamps = [0,12,24,36,48,60,72,84,96,108,120] # Hard coded time steps for the RNN testing data
    seq_length = 11 # Number of data points passed into network. 12th and 13th trimmed off due to some expts only having 11 data points
else:
    print("Invalid neural network type, choose either fnn or rnn")
    
expt_data, raw_testing_data, test_scaler = data_preprocess(network_type) 

for j in range(0,len(hyperparam_list)):
    results[j] = neural_network_evaluation(hyperparam_list[j], expt_data, test_scaler, cores, n_samples, network_type)
    avg_mse[j] = results[j][0]
    testing_set_data[j] = results[j][6]
    biomass_predictions[j] = results[j][7]
fin_time = datetime.now()

#print_results(hyperparam_list, results, network_type) # Func call, prints evaluation outputs such as the means and std devs
print("\nScript execution time: ", (fin_time-init_time), "(hrs:mins:secs)")
print("No. of parameter configurations: ", len(hyperparam_list))

make_plot = 0 # Prediction plots for the 5 testing set errors (correspond to closest_to_average value)
if make_plot == 1:
    #testing_expt = [3,7,9,18,22]
    #testing_expt = [7,9,18,22]
    testing_expt = [5, 10, 15, 20]
    sl = seq_length
    for i in range(0, len(hyperparam_list)):
        cfg = hyperparam_list[i]
        if network_type == 'fnn':
            window_title = 'Error_'+str(round(avg_mse[i],4))+'_HN_('+str(cfg[0])+'_'+str(cfg[1])+')_Epochs_'+str(cfg[2])+'_Batch_Size_'+str(cfg[3])+'_Learn_rate_'+str(cfg[4])
        if network_type == 'rnn':
            window_title = 'Error_'+str(round(avg_mse[i],4))+'_HN_'+str(cfg[0])+'_Epochs_'+str(cfg[1])+'_Batch_Size_'+str(cfg[2])+'_Learn_rate_'+str(cfg[3])
       
        for j in range(0,len(testing_expt)):
            biomass_experimental_data = raw_testing_data['BC'][(j*sl):sl+(j*sl)]
            predictions = biomass_predictions[i][(j*sl):sl+(j*sl)]
            fig, ax = plt.subplots()
            fig.canvas.set_window_title(window_title+'_expt'+str(testing_expt[j]))     

            line1, = ax.plot(timestamps, biomass_experimental_data, 'rx',label='experiment')
            line2, = ax.plot(timestamps, (predictions), 'b+',label='prediction')
            line3 = Line2D([0], [0], color='white', linewidth=0.1, linestyle='None')
            ax.legend((line1,line2,line3),('Experiment no. '+str(testing_expt[j]),'Prediction','Average error: '+str(round(testing_set_data[i][j],3))))
            ax.set_ylabel('Biomass concentration (g/L)')
            ax.set_xlabel('Time (hrs)')
            plt.minorticks_on()
            plt.grid(True)
            plt.show()
