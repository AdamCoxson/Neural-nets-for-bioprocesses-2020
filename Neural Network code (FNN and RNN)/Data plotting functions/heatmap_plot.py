# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 14:52:51 2020

Author: Adam Coxson, MPhys Undergrad, The University of Manchester
Project: Evaluating hyperparameter optimisation techinques for an FNN and RNN
Module: heatmap_plot
Dependancies: none

This script plots heat maps for analysis of either an FNN or RNN network
architecture used during this project. The data is obtain using grid search
hyperparamter tuning along 2 axes. The heat maps enable the tuning engineer to
gain an intuitive understanding of the features and trends within the domain 
space of the network being evaluated. This script contains contour plots
as well as some of the initial data processing.

This is left here for completeness and for you to see how the heatmaps were 
programmed.

"""

import csv, sys
import numpy as np
import matplotlib.pyplot as plt


def csv_data_reader(filename):
    data = []
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        next(reader) # these skip the two header lines
        next(reader)
        try:
            for row in reader:
                #if(reader.index(row) == 0 or reader.index(row)==1):
                 #   continue
                data.append(row)
        except csv.Error as e:
            sys.exit('file {}, line {}: {}'.format(filename, reader.line_num, e))
    return data

def get_mse_Zdata_v1(len_x,len_y,file_data):
    mse_data = np.zeros((len_x, len_y))
    k = 0
    for i in range(0,len_x):
        for j in range(0,len_y):
            mse_data[i][j] = (file_data[k][0])
            k=k+1
    return mse_data

def get_mse_Zdata_v2(len_x,len_y,file_data):
    mse_data = np.zeros((len_x, len_y))
    k = 0
    for i in range(0,len_x):
        for j in range(0,len_y):
            mse_data[i][j] = (file_data[k])
            k=k+1
    return mse_data

def contour_plot(xyz_data,labels,levs,cols):
    
    fig, axe = plt.subplots()
    axe.set_title(labels[0])
    axe.set_xlabel(labels[1])
    axe.set_ylabel(labels[2])
    plot1 = axe.contourf(xyz_data[0], xyz_data[1], xyz_data[2], levs, colors=cols)
    cb = plt.colorbar(plot1, ax=axe, extend='both')
    cb.set_label(labels[3])
    fig.tight_layout()
    return 0

def logY_contour_plot(xyz_data,labels,levs,cols):
 
    fig, axe = plt.subplots()
    axe.set_title(labels[0])
    axe.set_xlabel(labels[1])
    axe.set_ylabel(labels[2])
    axe.set_yscale('log')
    plot1 = axe.contourf(xyz_data[0], xyz_data[1], xyz_data[2], levs, colors=cols)
    cb = plt.colorbar(plot1, ax=axe, extend='both')
    cb.set_label(labels[3])
    fig.tight_layout()
    return 0

def RNN_hidden_neuron_analysis(filename, title, repeats, neurons, plot):
    HN_data = csv_data_reader(filename)
    avg_data = [0]*neurons
    for i in range(0,repeats):
        for j in range(0,neurons):
            avg_data[j] = avg_data[j] + (1/repeats)*float(HN_data[j+(i*neurons)][0])
    HN = np.linspace(1,neurons,neurons)
    if plot == True:
        plt.figure()
        #.set_window_title('Hidden neuron analysis')
        plt.plot(HN,avg_data)
        plt.title(title)
        plt.xticks([2,4,6,8,10,12,14,16,18,20])
        plt.xlabel('Hidden neurons')
        plt.ylabel('Averaged MSE')
        plt.minorticks_on()
        plt.grid(True)
        plt.show()
    return avg_data

def FNN_HN_plot(filename, levs, param_label, max_lim):
    """
    ARGS:
        filename,
        levs - values to define colour axis boundaries,
        param_label - partial part of the label to give details of hyperparameters used,
        max_lim - A max value to set for any outliers, keeps axis nice and readable.
    OUTPUTS:
        None - creates a plot to the console

    This function takes in data to plot a 2D heat map of neuron configurations
    between a first and second layer of a Feed-forward network used in this project.
    """
    avg_data = [0]*36
    HN_data = []
    HN_data = (csv_data_reader(filename))
    
    for i in range(0,10):
        for j in range(0,36):
            if(float(HN_data[j+(i*36)][0])>max_lim):
                avg_data[j] = avg_data[j] + max_lim
                #print("big (",i, ",",j,") avg_sum:", avg_data[j],", current val:",float(HN_data[j+(i*36)][0]))
            else:
                avg_data[j] = avg_data[j] + float(HN_data[j+(i*36)][0])
                #print("(",i, ",",j,") avg_sum:", avg_data[j],", current val:",float(HN_data[j+(i*36)][0]))
            
    H1 = H2 = [2, 4, 6, 8, 10, 12]
    cols = ['#f1eef6','#d0d1e6','#a6bddb','#74a9cf','#3690c0','#0570b0','#034e7b']
    #levs = [0.0315,0.032,0.0325,0.033,0.0335,0.034,0.0352]
    HN_transpose = np.array(HN_data).transpose()[0]
    for i in range(0, 10):
        labels = ['Plot '+(str(i+1))+param_label,'HN-1','HN-2','MSE']
        mse_data = get_mse_Zdata_v2(6,6, HN_transpose[0+(36*i):36+(36*i)]) 
        #contour_plot([H1, H2, mse_data], labels, levs, cols)
    k = 0
    for i in range(0,6):
        for j in range(0,6):
            mse_data[i][j] = (avg_data[k]/10)
            k=k+1
    labels = ['','Layer 1 neuron number','Layer 2 neuron number','MSE']
    # AVERAGED PLOT
    contour_plot([H1, H2, mse_data], labels, levs, cols) ##########################
    
    

""" ------------------------------------------------------------------------
FNN neuron number evaluations at 200 epochs and batch size of 80.
Neuron numbers varied from 2 to 12 for a two layer, sigmoid activation function.
Repeated for learning rates of 0.1, 0.01, 0.001, 0.0001.
Furthermore, heat maps for epochs versus learning rate and epochs versus
batch size, each at 51 value, are plotted here.
This is for the two-step grid search method I described in my report
"""
param_label = ', Epochs 200, Batch size 80, Learning rate 0.01'

filename = 'Grid_FNN_HNplot_lr0p1_2.csv' 
levs = [0.040,0.050,0.060,0.070,0.080,0.09,0.12]
FNN_HN_plot(filename, levs, param_label, 0.2)

filename = 'Grid_FNN_HNplot_lr0p01_2.csv'
levs = [0.04,0.05,0.06,0.07,0.08,0.09,0.11]
FNN_HN_plot(filename, levs, param_label, 0.5)

filename = 'Grid_FNN_HNplot_lr0p001_2.csv' 
levs = [0.036, 0.040, 0.044, 0.048, 0.052, 0.056, 0.062]
FNN_HN_plot(filename, levs, param_label, 0.5)

filename = 'Grid_FNN_HNplot_lr0p0001_2.csv' 
levs = [0.0315,0.032,0.0325,0.033,0.0335,0.034,0.0352]
FNN_HN_plot(filename, levs, param_label, 0.5)

# Epochs v batch size
EPOCHS = np.linspace(0,500,51)
BATCH_SIZE = np.linspace(0,500, 51)
LR = np.round(np.logspace(-4, -1, num = 51), 6)
EPOCHS[0] = BATCH_SIZE[0] = 1

data_BS_EPOCHS = csv_data_reader('Grid_FNN_51epochs_v_51bs_2.csv')
cols = ['#f1eef6','#d0d1e6','#a6bddb','#74a9cf','#3690c0','#0570b0','#034e7b']
levs = [0.020,0.035, 0.05, 0.065, 0.080, 0.1, 3]
levs = [0.020,0.030, 0.040, 0.050, 0.060, 0.080, 3]
# Learning rate = 0.01, HN = (8,4)
labels = ['','Epochs','Batch size','MSE']
mse_data = get_mse_Zdata_v1(51,51, data_BS_EPOCHS)
contour_plot([EPOCHS, BATCH_SIZE, mse_data], labels, levs, cols)

# Epochs v learning rate
data_LR_EPOCHS = csv_data_reader('Grid_FNN_51epochs_v_51lr_2.csv')
mse_data = get_mse_Zdata_v1(51, 51, data_LR_EPOCHS)
cols = ['#f1eef6','#d0d1e6','#a6bddb','#74a9cf','#3690c0','#0570b0','#034e7b']
#levs = [0.025,0.030,0.040, 0.05, 0.06, 0.08, 2]
levs = [0.020,0.035,0.050, 0.065, 0.08, 0.1, 3]
levs = [0.020,0.030, 0.040, 0.050, 0.060, 0.080, 3]
# Batch size 80, HN = (8,4)
labels = ['','Epochs','Learning rate','MSE']
#contour_plot([EPOCHS, LR, mse_data], labels, levs, cols)
logY_contour_plot([EPOCHS, LR, mse_data], labels, levs, cols)

""" ------------------------------------------------------------------------
RNN neuron number evaluations at 200 epochs and batch size of 80.
Neuron numbers varied from 1 to 20 for a single layer, tanh activation function.
Repeated for learning rates of 0.1, 0.01, 0.001, 0.0001.
Furthermore, heat maps for epochs versus learning rate and epochs versus
batch size, each at 51 value, are plotted here.

This is for the two-step grid search method I described in my report
"""

filenames = ['Grid_RNN_HN_0p1.csv','Grid_RNN_HN_0p01.csv','Grid_RNN_HN_0p001.csv','Grid_RNN_HN_0p0001.csv']
titles = ['Learning rate: 0.1','Learning rate: 0.01','Learning rate: 0.001','Learning rate: 0.0001']
avg_data = [0]*4
for i in range(0,4):
    avg_data[i] = RNN_hidden_neuron_analysis(filenames[i],titles[i], 10, 20, plot = False)
    
HN = np.linspace(1,20,20)
plt.figure()
for i in range(0,4):
    plt.plot(HN, avg_data[i], label=titles[i])
#plt.title(title)
plt.xlabel('Number of hidden neurons')
plt.xticks([2,4,6,8,10,12,14,16,18,20])
plt.ylabel('Averaged MSE')
plt.ylim(0.035,0.08)
plt.legend()
plt.minorticks_on()
plt.grid(True)
plt.show()

EPOCHS = np.linspace(0,500,51)
BATCH_SIZE = np.linspace(0,500, 51)
LR = np.round(np.logspace(-4, -1, num = 51), 6)
EPOCHS[0] = BATCH_SIZE[0] = 1

data_BS_EPOCHS = csv_data_reader('Grid_RNN_51epochs_v_51bs.csv')
cols = ['#f1eef6','#d0d1e6','#a6bddb','#74a9cf','#3690c0','#0570b0','#034e7b']
levs = [0.02,0.030,0.040, 0.050, 0.060, 0.07, 0.7]
# Learning rate = 0.01, HN = (8,4)
labels = ['','Epochs','Batch size','MSE']
mse_data = get_mse_Zdata_v1(51,51, data_BS_EPOCHS)
contour_plot([EPOCHS, BATCH_SIZE, mse_data], labels, levs, cols)


data_LR_EPOCHS = csv_data_reader('Grid_RNN_51epochs_v_51lr.csv')
mse_data = get_mse_Zdata_v1(51, 51, data_LR_EPOCHS)
cols = ['#f1eef6','#d0d1e6','#a6bddb','#74a9cf','#3690c0','#0570b0','#034e7b']
#levs = [0.019,0.03,0.035, 0.045, 0.050, 0.06, 0.85]
levs = [0.020,0.030,0.040, 0.050, 0.060, 0.070, 0.900]
# Batch size 80, HN = (8,4)
labels = ['','Epochs','Learning rate','MSE']
#contour_plot([EPOCHS, LR, mse_data], labels, levs, cols)
logY_contour_plot([EPOCHS, LR, mse_data], labels, levs, cols)

data_LR_EPOCHS = csv_data_reader('Grid_RNN_51epochs_v_51lr_10neurons.csv')
mse_data = get_mse_Zdata_v1(51, 51, data_LR_EPOCHS)
cols = ['#f1eef6','#d0d1e6','#a6bddb','#74a9cf','#3690c0','#0570b0','#034e7b']
#levs = [0.019,0.03,0.035, 0.045, 0.050, 0.06, 0.85]
levs = [0.025,0.035,0.045, 0.055, 0.065, 0.080, 0.4]
# Batch size 80, HN = (8,4)
labels = ['','Epochs','Learning rate','MSE']
#contour_plot([EPOCHS, LR, mse_data], labels, levs, cols)
logY_contour_plot([EPOCHS, LR, mse_data], labels, levs, cols)


