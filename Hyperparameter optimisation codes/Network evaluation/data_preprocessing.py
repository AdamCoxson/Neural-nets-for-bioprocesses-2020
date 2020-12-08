# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 16:35:40 2020
Author: Adam Coxson, MPhys Undergrad, The University of Manchester
Project: Evaluating hyperparameter optimisation techinques for Bioprocesses
"Finding the Optimal Optimiser"
Module: data_preprocessing.py
Uses the pre-processing for the training and testing data in Mostafizor
Rahmans thesis work. When this module is called it formats the data to be fed
straight into the main neural network training loop.
"""
import pandas as pd
import numpy as np 
from replicate import replicate_data 
from sklearn.preprocessing import StandardScaler

def data_preprocess(net_type):
    
    if (net_type == 'fnn'):
        seq_length = 12
    elif (net_type == 'rnn'):
        seq_length = 11
    
    temp = pd.read_excel('Experimental_data_v2.xlsx', sheet_name='Experimental data')
    data_init = temp[2:]
    data_Op = pd.read_excel('Experimental_data_v2.xlsx', sheet_name='Operating conditions')
    training_list = []
    testing_list = []
    #testing_set = [3,7,9,18,22]
    #testing_set = [7,9,18,22]
    testing_set = [5, 10, 15, 20]
    
    for i in range(1,26):
        test_set = False
        for ts in testing_set:
            if (ts == i):
                test_set = True
        for j in range(0,seq_length):
            if test_set == True:
                testing_list.append([data_init.iloc[j][i], data_Op.iloc[1][i], data_Op.iloc[2][i], data_Op.iloc[3][i]])
            else:
                training_list.append([data_init.iloc[j][i], data_Op.iloc[1][i], data_Op.iloc[2][i], data_Op.iloc[3][i]])
            
    # Convert training data to pd dataframe
    columns = "BC GA C02 LI ".split()
    training_data = pd.DataFrame(data=training_list, index=None, columns=columns)
    testing_data = pd.DataFrame(data=testing_list, index=None, columns=columns)
    raw_testing_data = testing_data # Raw data outputted for network prediction evaluation
        
    # Standardise training and testing data
    scaler_train = StandardScaler()
    scaler_test = StandardScaler()
    scaler_train.fit(training_data)
    scaler_test.fit(testing_data)
    testing_data = scaler_test.transform(testing_data)
    
    # Replicate the training data with 3% and 5% error seeding
    replicated_data1 = replicate_data(training_data, 20, 0.03)
    replicated_data2 = replicate_data(training_data, 20, 0.05)
    training_data = training_data.append(replicated_data1, ignore_index=True, sort=False)
    training_data = training_data.append(replicated_data2, ignore_index=True, sort=False)
    training_data = scaler_train.transform(training_data)
    training_data = np.array(training_data)
    
    # Calculate training and testing labels
    try:
        a = []
        for index, row in enumerate(training_data):
            dBC = training_data[index + 1][0] - row[0]
            a.append([dBC])
    except IndexError:
        a.append([0])
    
    a = np.array(a)
    training_data = np.append(training_data, a, axis=1)
    
    try:
        a = []
        for index, row in enumerate(testing_data):
            dBC = testing_data[index + 1][0] - row[0]
            a.append([dBC])
    except IndexError:
        a.append([0])
    
    a = np.array(a)
    testing_data = np.append(testing_data, a, axis=1)
    
    # Remove the 11th (RNN) or 12th (FNN) data point from training and testing sets
    count = 0
    decrement = 0
    for index, row in enumerate(training_data):
        count += 1
        if count == seq_length:
            delete = index - decrement
            training_data = np.delete(training_data, delete, 0)
            decrement += 1
            count = 0
    
    count = 0
    decrement = 0
    for index, row in enumerate(testing_data):
        count += 1
        if count == seq_length:
            delete = index - decrement
            testing_data = np.delete(testing_data, delete, 0)
            decrement += 1
            count = 0
    
    # Shuffle training data
    np.random.shuffle(training_data)
    training_inputs = training_data[:, 0:len(testing_set)]
    training_labels = training_data[:, len(testing_set):]
    test_inputs = testing_data[:, 0:len(testing_set)]
    test_labels = testing_data[:, len(testing_set):]
    
    if (net_type == 'rnn'):
            training_inputs = np.split(training_inputs, len(training_inputs)/(seq_length-1))
            training_labels = np.split(training_labels, len(training_labels)/(seq_length-1))
            test_inputs = np.split(test_inputs, len(testing_set))
            test_labels = np.split(test_labels, len(testing_set))
    
    return training_inputs, training_labels, test_inputs, test_labels, raw_testing_data, scaler_test

