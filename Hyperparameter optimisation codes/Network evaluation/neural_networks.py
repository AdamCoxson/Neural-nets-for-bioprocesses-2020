
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

torch.manual_seed(777)

class fnn_net(nn.Module):
	'''
	This Class Defines the Structure of the Artificial Neural Network
	'''
	def __init__(self, HN1, HN2):	
		self.HN1 = HN1			
		self.HN2 = HN2
		super().__init__()                # Run the intitialision method from base class nn.module.
        
		self.fc1 = nn.Linear(4, self.HN1) # Define the first fully connected layer. 
                                          # nn.Linear connects input nodes to output nodes in standard way.
                                          # The input layer contains 5 nodes. 
                                          # The output layer (first hidden layer), consists of 15 nodes.
                                                   
		self.fc2 = nn.Linear(self.HN1, self.HN2)   # Hidden layer 2: each node takes in 15 values, contains 15 nodes hence outputs 15 values.
        
		self.fc3 = nn.Linear(self.HN2, 1)          # Output Layer: each node takes in 15 values, 
                                                   # contain 3 nodes (one for each rate of change: X, N and Lu) hence outputs 3 values.
                                                   

	def forward(self, x):
        # This method feeds data into the network and propagates it forward. 
        # Feed the dataset, x, through fc1 and apply the Sigmoid activation 
        # function to the weighted sum of each neuron. Then assign the 
        # transformed dataset to x. Next, feed the transformed dataset through
        # fc2 and so on... until we reach the output layer. The activation 
        # function basically decides if the neuron is 'firing' like real 
        # neurones in the human brain. The activation function prevents massive
        # output numbers.
        
		x = torch.sigmoid(self.fc1(x))   
		x = torch.sigmoid(self.fc2(x))    
		x = self.fc3(x)
		return x 
    
# num_outputs = 1 # Number of outputs in output layer, if only using one cell with no liner layer, then this is equal to hidden_size
# input_size = 4  
# sequence_length = 10 # Number of datapoints per experimental run
# hidden_size = 4  # number of nodes in hidden state   
# num_layers = 2  # two-layer rnn

### Model ###
class rnn_net(nn.Module):
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
