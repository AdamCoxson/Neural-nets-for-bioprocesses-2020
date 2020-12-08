import torch
import torch.nn as nn
from torch.autograd import Variable

torch.manual_seed(777)

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
