# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 16:54:01 2020
Author: Adam Coxson, MPhys Undergrad, The University of Manchester
Project: Evaluating hyperparameter optimisation techinques for an FNN and RNN
Module: neural_networks 
Dependancies: none

This contains classes which are called in other scripts to initialise Feed-forward and 
recurrent neural networks. These network classes are made using the pyTorch package 
functionality. During the project the FNN was fixed at 2-layers and the sigmoid activation
function. The RNN was fixed at only 1-layer with the tanh activation function.

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
torch.manual_seed(777)

class fnn_net(nn.Module):
    '''
    This class defines the structure of the Feed-forward Neural Network.
    This FNN is fixed at 2 layers.
    '''
    def __init__(self, HN1, HN2):
        self.HN1 = HN1			
        self.HN2 = HN2
        super().__init__()                # Run the intitialision method from base class nn.module.
        
        num_inputs = 4   # Number of input variables. Biomass Conc. and 3 other initial conditions  (constants)
        num_outputs = 1  # Predicited difference in Biomass concentration for the next iteration
        
        self.fc1 = nn.Linear(num_inputs, self.HN1)      # First fully connected layer.                                                                       
        self.fc2 = nn.Linear(self.HN1, self.HN2)        # Second layer - iterate here for multi-layer variability
        self.fc3 = nn.Linear(self.HN2, num_outputs)     # Output layer    

    def forward(self, x):
        """
        This method feeds data into the network and propagates it forward. 
        Feed the dataset, x, through fc1 and apply the Sigmoid activation 
        function to the weighted sum of each neuron. Then assign the 
        transformed dataset to x. Next, feed the transformed dataset through
        fc2 and so on... until we reach the output layer. The activation 
        function decides if the neuron is 'firing' and prevents massive output numbers.
        """
        x = torch.sigmoid(self.fc1(x))   
        x = torch.sigmoid(self.fc2(x))    
        x = self.fc3(x)
        return x 
    
class rnn_net(nn.Module):
    '''
	This class defines the structure of the Recurrent Neural Network.
    num_outputs     = 1   # Number of outputs in output layer.
    input_size      = 4   # Number of input variables
    sequence_length = 10  # Number of datapoints per experimental run
    hidden_size     = 4   # number of nodes in hidden state   
	'''

    def __init__(self, num_outputs, input_size, sequence_length, hidden_size, num_layers):
        super(rnn_net, self).__init__()

        self.num_outputs = num_outputs
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.num_outputs)
    
    def forward(self, x, hidden):
        # Reshape input to (batch_size, sequence_length, input_size)
        x = x.view(x.size(0), self.sequence_length, self.input_size)

        # Propagate input through RNN
        # Input: (batch, seq_len, input_size)
        out, _ = self.rnn(x, hidden)
        fc_out = self.fc(out)
        return fc_out, _
    
    def init_hidden(self, x):
        # Initialse hidden and cell states
        # (num_layers * num_directions, batch_size, hidden_size)
        return Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
