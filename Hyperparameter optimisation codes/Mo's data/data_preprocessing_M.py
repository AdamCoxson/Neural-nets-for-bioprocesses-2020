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

def data_preprocess():

    # Load training and testing data as pd dataframe
    training_data = pd.read_excel('Data3/reduced_training_data.xlsx')
    testing_data = pd.read_excel('Data3/test_data.xlsx')
    
    # Standardise training and testing data
    scaler_train = StandardScaler()
    scaler_test = StandardScaler()
    
    scaler_train.fit(training_data)
    scaler_test.fit(testing_data)
    
    testing_data = scaler_test.transform(testing_data)
    
    # Convert training data to pd dataframe
    columns = "BC NC LP LI NIC".split()
    training_data = pd.DataFrame(data=training_data, index=None, columns=columns)
    
    # Replicate the training data
    replicated_data1 = replicate_data(training_data, 50, 0.03)
    replicated_data2 = replicate_data(training_data, 50, 0.05)
    
    training_data = training_data.append(replicated_data1, ignore_index=True, sort=False)
    training_data = training_data.append(replicated_data2, ignore_index=True, sort=False)
    
    training_data = scaler_train.transform(training_data)
    training_data = np.array(training_data)
    
    # Calculate training and testing labels
    try:
        a = []
        for index, row in enumerate(training_data):
            dBC = training_data[index + 1][0] - row[0]
            dNC = training_data[index + 1][1] - row[1]
            dLP = training_data[index + 1][2] - row[2]
            
            rates = [dBC, dNC, dLP]
            a.append(rates)
    except IndexError:
        rates = [0, 0, 0]
        a.append(rates)
    
    a = np.array(a)
    training_data = np.append(training_data, a, axis=1)
    
    try:
        a = []
        for index, row in enumerate(testing_data):
            dBC = testing_data[index + 1][0] - row[0]
            dNC = testing_data[index + 1][1] - row[1]
            dLP = testing_data[index + 1][2] - row[2]
            
            rates = [dBC, dNC, dLP]
            a.append(rates)
    except IndexError:
        rates = [0, 0, 0]
        a.append(rates)
    
    a = np.array(a)
    testing_data = np.append(testing_data, a, axis=1)
    
    # Remove all datapoints corresponding to 144 h from the training and testing sets
    count = 0
    decrement = 0
    for index, row in enumerate(training_data):
        count += 1
        if count == 13:
            delete = index - decrement
            training_data = np.delete(training_data, delete, 0)
            decrement += 1
            count = 0
    
    count = 0
    decrement = 0
    for index, row in enumerate(testing_data):
        count += 1
        if count == 13:
            delete = index - decrement
            testing_data = np.delete(testing_data, delete, 0)
            decrement += 1
            count = 0
    
    # Shuffle training data
    np.random.shuffle(training_data)
    
    return training_data, testing_data