# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 13:34:58 2020

@author: adamc
For evaluating potential candidates of the optimal hyper-parameters
"""

import numpy as np                   # Packages -----
import pandas as  pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import copy
import csv
import multiprocessing
from cpuinfo import get_cpu_info
from datetime import datetime
#from joblib.externals.loky import set_loky_pickler
#from joblib import parallel_backend
from joblib import Parallel, delayed
#from joblib import wrap_non_picklable_objects
from ann import Net                # Modules -----
from train_v2 import train
from test_module import test
from sklearn.preprocessing import StandardScaler
from data_preprocessing import data_preprocess

def closest_to_average(vals, avg):
    diff_from_avg = np.sqrt((vals - avg)**2)
    sorted_idx = np.argsort(diff_from_avg)
    #print("\nDiff:",diff_from_avg, "Idx:",sorted_idx,"\n")
    return sorted_idx[0]

def neural_net(cfg):
    n_samples = 50
    start_time = datetime.now()
    h1     = int(cfg[0])
    h2     = int(cfg[1])
    EPOCHS = int(cfg[2])
    bs     = int(cfg[3])
    lr     = float(cfg[4])
    mse_vals = np.zeros(n_samples)
    prediction_data = [0]*n_samples
    testing_set_data = [0]*n_samples
    


    for i in range(0,n_samples):
        net = Net(h1, h2)
        init_state = copy.deepcopy(net.state_dict())
        net.load_state_dict(init_state)
        train(net, training_inputs, training_labels, EPOCHS, lr, bs)
        mse_vals[i], testing_set_data[i], prediction_data[i] = test(test_inputs, test_labels, net)
    avg_error = np.mean(np.sqrt(mse_vals))
    std_error = np.std(np.sqrt(mse_vals))
    
    avg_idx = closest_to_average(mse_vals, np.mean(mse_vals))
    closest_error = np.sqrt(mse_vals[avg_idx]) 
    testing_set = np.sqrt(np.array(testing_set_data[avg_idx]))
    predictions_inverse = scaler_test.inverse_transform(prediction_data[avg_idx])
    prediction = predictions_inverse.T[0][:]
    end_time = datetime.now()
    #print('\nParameters HN: (%d, %d), Epochs: %d, Batch Size: %d, Learn rate: %.4f'%(cfg[0],cfg[1],cfg[2],cfg[3],cfg[4]))
    #print("MSE values:\n",(mse_vals))
    #print('Mean Squared Error: %.4f \u00b1 %.4f, Mean Error: %.4f \u00b1 %.4f'%(avg_error**2,np.std(mse_vals),avg_error,std_error))
    #print("FNN training execution time: ", (end_time-start_time), "(hrs:mins:secs)")
    return (avg_error, closest_error, testing_set, prediction, mse_vals, (end_time-start_time))

def print_results(hyperparam_list, results):
    for i in range(0,len(hyperparam_list)):
        cfg = hyperparam_list[i]
        mse_vals = results[i][4]
        time = results[i][5]
        mean_MSE = np.mean(mse_vals)
        std_MSE = np.std(mse_vals)
        mean_error = np.mean(np.sqrt(mse_vals))
        std_error = np.std(np.sqrt(mse_vals))
        print('\nParameters HN: (%d, %d), Epochs: %d, Batch Size: %d, Learn rate: %.4f'%(cfg[0],cfg[1],cfg[2],cfg[3],cfg[4]))
        print("MSE values:\n",(mse_vals))
        print('Mean Squared Error: (%.4f \u00b1 %.4f), Mean Error: (%.4f \u00b1 %.4f)'%(mean_MSE,std_MSE,mean_error,std_error))
        print("FNN training execution time: ", time, "(hrs:mins:secs)")
    

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
training_inputs = training_data[:, 0:4]
training_labels = training_data[:, 4:]
test_inputs = testing_data[:, 0:4]
test_labels = testing_data[:, 4:]

# np.mean([0.0744,0.0567,0.0742, 0.0811,0.0469,0.0914, 0.0822, 0.0481, 0.0668, 0.0412, 0.0677, 0.0522])
# H1, H2, Epochs, Batch size, learning rate. These parameters are for the FNN
hyperparam_list = [[2,8,71,106,0.0114],[6,2,360,178,0.0110],[2,8,318,89,0.0701],[10,8,334,31,0.1], # Bayesian
                  [6,8,52,256,0.0515],[4,8,119,265,0.0912],[4,6,355,157,0.0858],[4,10,59,184,0.0465], # Random Forest
                  [4,4,150,10,0.0002], [4,4,300,200,0.0004],[4,4,300,300,0.0004],[6,4,10,50,0.004],[4,2,50,50,0.001], # Grid search
                  [4,2,389,121,0.0003],[4,4,256,245,0.0004],[4,4,52,234,0.0017],[6,4,113,73,0.0004],[4,2,366,239,0.0005]] # Random search
#hyperparam_list = [[4,8,125,50,0.1], [4,8,30,10,0.01]]
#hyperparam_list = [[6,2,360,178,0.0110],[6,2,360,178,0.0110],[6,2,360,178,0.0110],[6,2,360,178,0.0110],[6,2,360,178,0.0110],
#                   [6,2,360,178,0.0110],[6,2,360,178,0.0110],[6,2,360,178,0.0110],[6,2,360,178,0.0110],[6,2,360,178,0.0110]]
#hyperparam_list = [[2,8,71,106,0.0114]]
biomass_predictions = [0]*len(hyperparam_list)
avg_mse = [0]*len(hyperparam_list)
closest_error = [0]*len(hyperparam_list)
testing_set_data = [0]*len(hyperparam_list)
results = [0]*len(hyperparam_list)


cores = 4
init_time = datetime.now()
print("Parallel processing activated, print functions may be surpressed.")
results[:] = Parallel(n_jobs=cores)(delayed(neural_net)(config) for config in hyperparam_list)
for i in range(0,len(avg_mse)):
    avg_mse[i] = results[i][0]
    closest_error[i] = results[i][1]
    testing_set_data[i] = results[i][2]
    biomass_predictions[i] = results[i][3]
fin_time = datetime.now()

print_results(hyperparam_list, results)
print("\nScript execution time: ", (fin_time-init_time), "(hrs:mins:secs)")
print("No. of parameter configurations: ", len(hyperparam_list) )

make_plot = 0
if make_plot == 1:
    timestamps = [0,12,24,36,48,60,72,84,96,108,120,132]
    testing_expt = [3,7,9,18,22]
    for i in range(0, len(hyperparam_list)):
        for j in range(0,5):
            biomass_experimental_data = raw_testing_data['BC'][(j*12):12+(j*12)]
            predictions = biomass_predictions[i][(j*12):12+(j*12)]
            cfg = hyperparam_list[i]
            fig, ax = plt.subplots()
            fig.canvas.set_window_title('Error: %.4f, HN:(%d, %d), Epochs:%d, Batch Size %d, Learn rate: %.4f'%(avg_mse[i],
                                                                                    cfg[0],cfg[1],cfg[2],cfg[3],cfg[4]))
            line1, = ax.plot(timestamps, biomass_experimental_data, 'rx',label='experiment')
            line2, = ax.plot(timestamps, predictions, 'b+',label='prediction')
            line3 = Line2D([0], [0], color='white', linewidth=0.1, linestyle='None')
            #ax.text(0.5, 0.95, ('Average error: '+str(0.05)), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
            ax.legend((line1,line2,line3),('Experiment '+str(testing_expt[j]),'Prediction','Average error: '+str(round(testing_set_data[i][j],3))))
            ax.set_ylabel('Biomass concentration (g/L)')
            ax.set_xlabel('Time (hrs)')
            plt.minorticks_on()
            plt.grid(True)
            plt.show()




        

#filename = 'Grid_HNplot_epochs200_bs8_lr0p01.csv'
#write_to_csv(filename, avg_mse_data, hyperparam_list, ["H1","H2","Epochs","Batch size","Learning rate"],fin_time-init_time, cores)