# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 16:35:40 2020
Author: Adam Coxson, MPhys Undergrad, The University of Manchester
Project: Evaluating hyperparameter optimisation techinques for an FNN and RNN
Module: data_preprocessing.py
Dependancies: none

Based on work by Mostafizor Rahman, see README.txt

This script reads in experimental data from a formatted csv file. Some of the 
available data is selected for testing data and the rest is used as training
data. The script can then process the data for the FNN and RNN architectures 
used in this project. The processing involves normalisation and standard 
scaling of each input. The training data is reeplicated 20 times with a 
seeding of 3% random noise and another 20 times with a seeding of 5% random 
noise. Finally, the last data point from each experimental trial is removed
and the data formatted into the final set of lists. In this proejct the FNN 
used 11 data points and the RNN used 10 out of a total of 12 data points per 
trial, see the README.txt for more details.

If a different data set needs to be used then create a 2nd, or an improved
version of this script to handle the new data input format and sequence length.
Current input variables, 1 varying and 3 constants
Varying variable - biomass concentration g/L,
Constant - Gas Aeration rate GA
Constant - Carbon Dioxide concentration CO2
Constant- Light intensity LI
"""

# Package imports
import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
 

def replicate_data(data, replications, noise):           
    """
    ARGS: 
        data: the continuous list of data to be replicated
        replications: the number of data copies to make
        noise: the noise fraction to seed the replicated data with
    OUTPUTS:
        new_data: The list of data after replication and noise seeding
    """          
    cols = list(data.columns) 
    dataR = data[cols[0]]                                
    df = data                                              
    new_data = pd.DataFrame(columns=data.columns)
    i = 0                                                                                      
    while i < replications:
        replicated_data =  np.random.uniform(dataR-dataR*noise, dataR+dataR*noise) # Seed data with random noise              
        replicated_data = pd.DataFrame(data=replicated_data, index=None, columns=['BC'])  # Reformat replicated data
        replicated_data['GA'] = df[cols[1]]         # Add back in the initial conditions/constant variables                                             
        replicated_data['CO2'] = df[cols[2]]
        replicated_data['LI'] = df[cols[3]]                                                     
        new_data = new_data.append(replicated_data, ignore_index=True, sort=False) # Combine new and old data
        i += 1
    return new_data

def data_preprocess(net_type):
    """
    ARGS:    
        network type (either fnn or rnn).
    OUTPUTS: 
        training_inputs: return normalised and formatted training data,
        training_labels: return normalised and formatted training data,
        test_inputs: return normalised and formatted testing data,
        test_labels: return normalised and formatted testing data,
        raw_testing_data: return the raw, non-normalised testing data for prediction plots
        plots as part of network output evaluation,
        scaler_test: scaled testing data for formatting prediction data in evalution script.
    
    This function reads in a CSV from the cwd which contains formatted data for a bioprocess.
    The data is then organised into training and testing data and normalised for use
    in the FNN or the RNN.
    """
    if (net_type == 'fnn'):
        seq_length = 12
    elif (net_type == 'rnn'):
        seq_length = 11
        
    # Initial data reading and formatting
    temp = pd.read_excel('Experimental_data_v2.xlsx', sheet_name='Experimental data')
    data_init = temp[2:]
    data_Op = pd.read_excel('Experimental_data_v2.xlsx', sheet_name='Operating conditions')
    training_list = []
    testing_list = []
    #testing_set = [3,7,9,18,22] # Selected testing data
    #testing_set = [7,9,18,22]
    testing_set = [5, 10, 15, 20]
    
    for i in range(1,26): # Forming training and testing data into one long, continous list
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
    columns = "BC GA CO2 LI ".split()
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
        temp = []
        for index, row in enumerate(training_data):
            dBC = training_data[index + 1][0] - row[0] # dBC, difference in biomass between data points
            temp.append([dBC])                         # Add in varying inputs here
    except IndexError:
        temp.append([0])
    temp = np.array(temp)
    training_data = np.append(training_data, temp, axis=1)
    
    try:
        temp = []
        for index, row in enumerate(testing_data):
            dBC = testing_data[index + 1][0] - row[0]
            temp.append([dBC])
    except IndexError:
        temp.append([0])
    
    temp = np.array(temp)
    testing_data = np.append(testing_data, temp, axis=1)
    
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
    
    # Shuffle training data and format final list
    np.random.shuffle(training_data)
    training_inputs = training_data[:, 0:len(testing_set)]
    training_labels = training_data[:, len(testing_set):]
    test_inputs = testing_data[:, 0:len(testing_set)]
    test_labels = testing_data[:, len(testing_set):]
    
    if (net_type == 'rnn'): # Split up the data into respective trial sets
            training_inputs = np.split(training_inputs, len(training_inputs)/(seq_length-1))
            training_labels = np.split(training_labels, len(training_labels)/(seq_length-1))
            test_inputs = np.split(test_inputs, len(testing_set))
            test_labels = np.split(test_labels, len(testing_set))
            
    data_output = [training_inputs, training_labels, test_inputs, test_labels]
    
    return data_output, raw_testing_data, scaler_test


