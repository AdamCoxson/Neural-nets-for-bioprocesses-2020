# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 13:34:58 2020
Author: Adam Coxson, MPhys Undergrad, The University of Manchester
Project: Evaluating hyperparameter optimisation techinques for an FNN and RNN
Module: HPO_network_evaluations 
Dependancies: neural_networks, data_preprocessing, FNN_test_module,
              RNN_test_module, train_v2
              
This script is for evaluating sets of optimal hyperparameters which have been
found previously by different hyperparameter optimisation techniques. This 
script allows for evaluating these sets multiple times to average the data and 
identifies any anomalous outliers which were flagged as optimal due to having
extremely high variance (such as MSE = 0.1 ± 0.08). Parallel processing is used
to speed up the execution time for a high number of samples. For effective
evaluation of mean and standard deviation, use n_samples = 100. For faster 
processing use n_samples = 20, not as accurate but is large enough to get 
within 1 std, usually. This can be used for the FNN and RNN architectures used 
in this project.

The Hyperparameters are defined by the user in MAIN and the program outputs the
results to the console as well as produce prediction plots for each cnofiguration.

UPDATE: 17/08/2020 - After my meeting with Dongda we concluded that the 
optimal configurations obtained from each of the different runing methods were
indeed somehwat optimised. However, using higher epoch numbers should enable
the networks to access higher levels of acuracy. Increasing the epoch number
can result in step like increases in accuracy after certain intervals.
(i.e. 100 epochs gives 0.5 error, 300 would start to give 0.4, 600 would 
give 0.2 and so forth). From here on, all scripts and data identifed as fine
tuning was my attempt to access greater accuracy by significantly increasing
the epoch number and optimising the respective learning rate manually.

PLAN:
Investigate (4,2) LR 0.001 for higher epoch and higher batch size numbers. I'm
hoping a very high batch size will scale with high epochs and enable me to 
access greater levels of accuracy.
For the RNN, I need to look at lower Epoch numbers, Mo's optimal value was 30
whereas I only properly invesitaged 40 to 400. Maybe more optimised sets lie
within the range of 1 to 100 for the RNN? This would drastically increase 
execution times too.
"""

import numpy as np                   # Packages -----
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import copy
from datetime import datetime
from joblib import Parallel, delayed
import psutil # For joblib worker timeout/memory leak

from neural_networks import fnn_net, rnn_net              # Modules -----
from train_v2 import train
from FNN_test_module import FNN_test
from RNN_test_module import RNN_test
from data_preprocessing import data_preprocess

def closest_to_average(vals, avg):
    """
    ARGS: (A list of values, the average value of that list)
    OUTPUTS: (The index of the value in the 'vals' list which is closest to 'avg').
    
    This function loops through 'vals' and find the value which is closest to val, then returns this.
    This is used to help me select predicition data which is most representative of the hyperparameters.
    """
    diff_from_avg = np.sqrt((vals - avg)**2)
    sorted_idx = np.argsort(diff_from_avg)
    #print("\nDiff:",diff_from_avg, "Idx:",sorted_idx,"\n")
    return sorted_idx[0]

def parallel_network_evaluation(EPOCHS, lr, bs, network_type,net):
    """
    ARGS:    (Epoch number, learning rate, batch size, network tpye (either fnn or rnn), network to be evaluated).
    OUTPUTS: (MSE value, 5 mse values from 5 data sets used as testing data, predictions for the 5 test sets).
    
    This function takes in a set of hyperparameters and evaluates them for a neural network. This is built to
    be used with joblibs 'embarrasingly parallel' parallel procedure. This calls the training and testing
    functions for either the fnn or rnn.
    """
    training_mse = train(net, training_inputs, training_labels, EPOCHS, lr, bs, network_type)
    if network_type == 'fnn':        
        mse_val, testing_set_data, prediction_data = FNN_test(test_inputs, test_labels, net)
    elif network_type == 'rnn':
        mse_val, testing_set_data, prediction_data = RNN_test(test_inputs, test_labels, net)
    else:
        print("Invalid neural network type, choose either fnn or rnn")
    return (mse_val, training_mse, testing_set_data, prediction_data)
    #return (mse_val, testing_set_data, prediction_data)
        
        

def FNN_neural_net(cfg, cores, n_samples):
    """
    ARGS: cfg: A single configuration of hyperparameters,
          cores: number of cores to be used in parallel processing,
          n_samples: number of samples to average over for a given set of hyperparameters.
          
    OUTPUTS: avg_error: The MSE value averaged from multiple iterations of training and testing,
             testing_set: the testing data whose average MSE is closest to Avg_error,
             prediction: the prediction data that corresponds to testing_set,
             mse_vals: The whole list of mse values calculated for averaging,
             training_mse_vals: The whole list of training data MSE values,
             Execution time.
             
    This function takes in a set of hyperparameters and evaluates them for the FNN architecture developed in the program. 
    It repeats the network training and testing n_samples times over and averages this data, utiliting parallel
    processing and the parallel_network_evaluation() function. This is usesd to further evaluate optimal hyperparameters
    identifed from previous analyse. The purpose of this is to identify any outliers wrongly believed to be optimal as 
    many hyperparameter sets have a very high variance after training and testing. The parallel processing used is from 
    the joblib package which automatically deals with multiple processes and the data reorganisation after processing.
    """
    start_time = datetime.now()
    h1     = int(cfg[0])
    h2     = int(cfg[1])
    EPOCHS = int(cfg[2])
    bs     = int(cfg[3])
    lr     = float(cfg[4])
    mse_vals = np.zeros(n_samples)
    training_mse_vals = [0]*n_samples
    results = [0]*n_samples
    networks = [0]*n_samples
    
    for i in range(0,n_samples):
        net = fnn_net(h1, h2)
        init_state = copy.deepcopy(net.state_dict())
        net.load_state_dict(init_state)
        networks[i] = net
    # Joblib parallelisation across the number of samples
    results[:] = Parallel(n_jobs=cores)(delayed(parallel_network_evaluation)(EPOCHS, lr, bs, 'fnn',net) for net in networks)
    for i in range (0,n_samples):
        mse_vals[i] = results[i][0]
        training_mse_vals[i] = np.array(float(results[i][1]))

    avg_error = np.mean(np.sqrt(mse_vals))
    avg_idx = closest_to_average(mse_vals, np.mean(mse_vals))
    testing_set = np.sqrt(np.array(results[avg_idx][2]))
    predictions_inverse = scaler_test.inverse_transform(results[avg_idx][3])
    prediction = predictions_inverse.T[0][:]
    end_time = datetime.now()
    return (avg_error, testing_set, prediction, mse_vals, training_mse_vals, (end_time-start_time))

def RNN_neural_net(cfg, cores, n_samples):
    """
    ARGS: cfg: A single configuration of hyperparameters,
          cores: number of cores to be used in parallel processing,
          n_samples: number of samples to average over for a given set of hyperparameters.
          
    OUTPUTS: avg_error: The MSE value averaged from multiple iterations of training and testing,
             testing_set: the testing data whose average MSE is closest to Avg_error,
             prediction: the prediction data that corresponds to testing_set,
             mse_vals: The whole list of mse values calculated for averaging, 
             training_mse_vals: The whole list of training data MSE values,
             Execution time.
             
    This function takes in a set of hyperparameters and evaluates them for the FNN architecture developed in the program. 
    It repeats the network training and testing n_samples times over and averages this data, utiliting parallel
    processing and the parallel_network_evaluation() function. This is usesd to further evaluate optimal hyperparameters
    identifed from previous analyse. The purpose of this is to identify any outliers wrongly believed to be optimal as 
    many hyperparameter sets have a very high variance after training and testing. The parallel processing used is from 
    the joblib package which automatically deals with multiple processes and the data reorganisation after processing.
    """
    start_time = datetime.now()
    HL = 1
    HN    = int(cfg[0])
    EPOCHS = int(cfg[1])
    bs     = int(cfg[2])
    lr     = float(cfg[3])
    mse_vals = np.zeros(n_samples)
    training_mse_vals = [0]*n_samples
    results = [0]*n_samples
    networks = [0]*n_samples

    
    for i in range(0,n_samples):
        net = rnn_net(1, 4, 10, HN, HL)
        init_state = copy.deepcopy(net.state_dict())
        net.load_state_dict(init_state)
        networks[i] = net
    # Joblib parallelisation across the number of samples and different instances of the same network
    results[:] = Parallel(n_jobs=cores)(delayed(parallel_network_evaluation)(EPOCHS, lr, bs, 'rnn',net) for net in networks)
    for i in range (0,n_samples):
        mse_vals[i] = results[i][0]
        training_mse_vals[i] = np.array(float(results[i][1]))

    avg_error = np.mean(np.sqrt(mse_vals))
    avg_idx = closest_to_average(mse_vals, np.mean(mse_vals))
    testing_set = np.sqrt(np.array(results[avg_idx][2]))
    predictions_inverse = scaler_test.inverse_transform(results[avg_idx][3])
    prediction = predictions_inverse.T[0][:]
    end_time = datetime.now()
    return (avg_error, testing_set, prediction, mse_vals, training_mse_vals, (end_time-start_time))

def print_results(hyperparam_list, results):
    """
    ARGS: hyperparam_list: a list of the 4/5 hyperparameters used,
          results: the averages, mse data and testing data results corresponiding to the hyperparameters: (avg_mse,
    closest error, testing set, prediction data, mse_vals and runtime).
    OUTPUTS: (no function return, outputs text to console for Copy pasta).
    
    This function displays the results from evaluation of optimal hyperparameters obtained from previous analysis.
    The outputs are the hyperparameters with the corresponding mse vals for each of the n_samples. It also gives
    the mean and standard deviation for the absolute mean error and the mean squared error.
    """
    for i in range(0,len(hyperparam_list)):
        cfg = hyperparam_list[i]
        mse_vals = results[i][3]
        training_vals = results[i][4]
        time = results[i][5]
        mean_MSE = np.mean(mse_vals)
        std_MSE = np.std(mse_vals)
        mean_error = np.mean(np.sqrt(mse_vals))
        std_error = np.std(np.sqrt(mse_vals))
        train_MSE = np.mean(training_vals)
        train_std = np.std(training_vals)
        if network_type == 'fnn':
            print('\nParameters HN: (%d, %d), Epochs: %d, Batch Size: %d, Learn rate: %.6f'%(cfg[0],cfg[1],cfg[2],cfg[3],cfg[4]))
        if network_type == 'rnn':
            print('\nParameters HN: %d, Epochs: %d, Batch Size: %d, Learn rate: %.6f'%(cfg[0],cfg[1],cfg[2],cfg[3]))
        print("MSE values:\n",(mse_vals))
        print('Mean Squared Error: (%.4f \u00b1 %.4f), Mean Error: (%.4f \u00b1 %.4f)'%(mean_MSE,std_MSE,mean_error,std_error))
        print('Training Mean Squared Error: (%.4f \u00b1 %.4f),)'%(train_MSE,train_std))
        print("Training execution time: ", time, "(hrs:mins:secs)")


##### MAIN ######
"""
FNN hyperparameter lists
These lists are obtained from the Optimal FNN results v2 text file found in Hyperparameter Optimisation\FNN.
HN1, HN2, Epochs, Batch size, learning rate. These parameters are for the FNN.
"""
# np.mean([0.0744,0.0567,0.0742, 0.0811,0.0469,0.0914, 0.0822, 0.0481, 0.0668, 0.0412, 0.0677, 0.0522])
#hyperparam_list = [[4,4,150,10,0.0002], [4,4,300,200,0.0004],[4,4,300,300,0.0004],[6,4,10,50,0.004],[4,2,50,50,0.001]] # Grid search
#hyperparam_list = [[4,2,389,121,0.0003],[4,4,256,245,0.0004],[4,4,52,234,0.0017],[6,4,113,73,0.0004],[4,2,366,239,0.0005],
#                   [6,8,52,256,0.0515],[4,8,119,265,0.0912],[4,6,355,157,0.0858],[4,10,59,184,0.0465]] # random search
#hyperparam_list = [[6,8,52,256,0.0515],[4,8,119,265,0.0912],[4,6,355,157,0.0858],[4,10,59,184,0.0465]] # Random forest
#hyperparam_list = [[4,8,40,105,0.0782],[4,2,133,143,0.0008],[4,8,60,87,0.0804],[4,10,226,162,0.0498]] # Bayesian New
#hyperparam_list = [[2,8,71,106,0.0114],[6,2,360,178,0.0110],[2,8,318,89,0.0701],[10,8,334,31,0.1]] # Bayesian Old
#hyperparam_list = [[10,8,334,31,0.1]]

# The list below is the optimal set from each method in order: Grid, Random search, Random Forest, Bayesian.
#hyperparam_list = [[4,2,50,50,0.001],[6,4,113,73,0.0004],[4,10,59,184,0.0465],[4,2,133,143,0.0008]] # Optimal

# FINE TUNING
#hyperparam_list = [[4,2,300,50,0.001],[6,4,300,73,0.0004],[4,10,200,184,0.0465],[4,2,300,143,0.0008]]
#hyperparam_list = [[4,2,300,100,0.006],[6,4,300,73,0.0004],[4,2,300,100,0.0008]]
#hyperparam_list = [[4,2,200,50,0.005],[6,4,500,100,0.001],[4,2,400,100,0.001]]
#hyperparam_list = [[4,2,50,50,0.001],[4,2,100,50,0.001]]
#hyperparam_list = [[4,2,200,50,0.001]]
#hyperparam_list = [[4,2,425,300,0.001],[4,2,350,300,0.001],[4,2,200,300,0.001],[4,2,175,300,0.001]]
#hyperparam_list = [[6,2,586,400,0.000311],[4,4,800,100,0.0001],[8,4,544,200,0.000149]] # A great set
# NEW OPTIMAL SET for the FNN
#hyperparam_list = [[8,4,186,500,0.0001], [8,4,186,500,0.0001]]
#hyperparam_list = [[8,4,186,20,0.0001],[8,4,141,456,0.0010], [8,4,245,55,0.00015],[8,4,510,200,0.00015],[6,2,586,400,0.000311],[4,4,800,100,0.0001]]

#network_type = 'fnn'
"""
RNN hyperparameter lists
These lists are obtained from the Optimal RNN results text file found in Hyperparameter Optimisation\RNN
HN, Epochs, Batch size, learning rate. These parameters are for the RNN.
"""
#hyperparam_list = [[2,181,127,0.0412],[2,120,37,0.0176],[2,237,139,0.0377],[7,329,30,0.0057]] # Bayesian
#hyperparam_list = [[2,260,195,0.0318],[9,381,135,0.0003],[2,252,105,0.0301],[9,348,82,0.0989]] # Random Forest 
#hyperparam_list = [[4,25,200,0.1],[2,50,250,0.1],[6,350,10,0.0001],[6,10,80,0.1],[2,25,50,0.01]] # Grid search
#hyperparam_list = [[4,264,166,0.0013],[3,43,229,0.0669],[5,295,195,0.0013],[3,382,122,0.0533],[2,96,210,0.0933]] # Random Search
#hyperparam_list = [[4,100,50,0.0001],[4,100,50,0.001]]
#hyperparam_list = [[2,234,137,0.02],[2,282,140,0.0361],[2,349,267,0.0404]] # RNN Bayesian v2
#hyperparam_list = [[9,381,135,0.0003]]

# The list below is the optimal set from each method in order: Grid, Random search, Random Forest, Bayesian.
#hyperparam_list = [[6,350,10,0.0001],[4,264,166,0.0013],[9,381,135,0.0003],[2,120,37,0.0176]] # Optimal
# The networks below with HN = 2 both seem to give decent predictions compared to the others
#hyperparam_list = [[2,200,377,0.007053],[2,200,378,0.006503],[8,65,366,0.011437],[14,49,500,0.02],[5,130,174,0.002675],[13,45,500,0.02]]

hyperparam_list = [[10,50,121,0.002913],[23,96,103,0.000762],[25,36,1,0.000106],[30,141,77,0.000443]]

network_type = 'rnn'

biomass_predictions = [0]*len(hyperparam_list) # list initialisation
avg_mse = [0]*len(hyperparam_list)
closest_error = [0]*len(hyperparam_list)
testing_set_data = [0]*len(hyperparam_list)
results = [0]*len(hyperparam_list)

#network_type = 'fnn'
cores = 4                  # Number of cores for multiprocessing.
n_samples = 100            # Number of evaluations per hyper-parameter set. For results I recommend 100, or at least 20
init_time = datetime.now()
print("Parallel processing activated, print functions are surpressed.")
# This section obtains the training data and runs the neural networks through the hyperparameters.
if network_type == 'fnn': 
    timestamps = [0,12,24,36,48,60,72,84,96,108,120,132] # Hard coded time steps for the FNN data
    seq_length = 12 # Number of data points passed into network. 13th is trimmed off due to some expts only having 11 data points
    
    training_inputs, training_labels, test_inputs, test_labels, raw_testing_data, scaler_test = data_preprocess('fnn') 
    # training_inputs = training_data[:, 0:4]
    # training_labels = training_data[:, 4:]
    # test_inputs = testing_data[:, 0:4]
    # test_labels = testing_data[:, 4:]
    
    for j in range(0,len(hyperparam_list)):
        results[j] = FNN_neural_net(hyperparam_list[j], cores, n_samples)
    
elif network_type == 'rnn':
    timestamps = [0,12,24,36,48,60,72,84,96,108,120] # Hard coded time steps for the RNN data
    seq_length = 11 # Number of data points passed into network. 12th and 13th trimmed off due to some expts only having 11 data points
    training_inputs, training_labels, test_inputs, test_labels, raw_testing_data, scaler_test = data_preprocess('rnn')
    
    #training_data, testing_data, raw_testing_data, scaler_test = data_preprocess('rnn')
    # training_inputs = training_data[:,0:4]
    # training_labels = training_data[:,4:]
    # test_inputs = testing_data[:,0:4]
    # test_labels = testing_data[:,4:]
    # training_inputs = np.split(training_inputs, len(training_inputs)/(seq_length-1))
    # training_labels = np.split(training_labels, len(training_inputs)/(seq_length-1))
    # test_inputs = np.split(test_inputs, 4)
    # test_labels = np.split(test_labels, 4)
    
    for j in range(0,len(hyperparam_list)):
        results[j] = RNN_neural_net(hyperparam_list[j], cores, n_samples)
else:
    print("Invalid neural network type, choose either fnn or rnn")
for i in range(0,len(avg_mse)):  # Collating data in large lists for further analysis
    avg_mse[i] = results[i][0]
    testing_set_data[i] = results[i][1]
    biomass_predictions[i] = results[i][2]
fin_time = datetime.now()
print_results(hyperparam_list, results) # Func call, prints evaluation outputs such as the means and std devs
print("\nScript execution time: ", (fin_time-init_time), "(hrs:mins:secs)")
print("No. of parameter configurations: ", len(hyperparam_list))

make_plot = 1 # Prediction plots for the 5 testing set errors (correspond to closest_error value)
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

#filename = 'Grid_HNplot_epochs200_bs8_lr0p01.csv'
#write_to_csv(filename, avg_mse_data, hyperparam_list, ["H1","H2","Epochs","Batch size","Learning rate"],fin_time-init_time, cores)