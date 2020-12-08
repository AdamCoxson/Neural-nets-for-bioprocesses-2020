import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import os
import sys

def train(net, inputs, labels, EPOCHS, l_rate, BATCH_SIZE, net_type):
    # create a text trap and redirect stdout
    sys.stderr = open(os.devnull, "w")
    net.train()
    optimiser = optim.Adam(net.parameters(), lr = l_rate) 
    # net.parameters():  all of the adjustable parameters in our network. 
    # lr: a hyperparameter adjusts the size of the step that the optimizer will take to minimise the loss.
    loss_function = nn.MSELoss(reduction='mean')
    loss_sum_function = nn.MSELoss(reduction='sum')
    loss_sum =  0
    rnn_seq_length = 10 # For the RNN: The number of data points for each training set (HARDCODED)
    if (net_type == 'fnn'):
        X = torch.Tensor(inputs).view(-1, 4)
        y = torch.Tensor(labels)
        for epoch in range(EPOCHS):
            for i in tqdm(range(0, len(X), BATCH_SIZE)):
                batch_X = X[i:i+BATCH_SIZE].view(-1, 4)
                batch_y = y[i:i+BATCH_SIZE]
                optimiser.zero_grad()
                outputs = net(batch_X)
                loss = loss_function(outputs, batch_y)
                loss_sum = loss_sum + loss_sum_function(outputs, batch_y)
                loss.backward()
                optimiser.step()  
                

    elif (net_type == 'rnn'):
        X = Variable(torch.Tensor(inputs))
        y = Variable(torch.Tensor(labels))
        for epoch in range(EPOCHS):
            for i in tqdm(range(0, len(X), BATCH_SIZE)):
                batch_X = X[i:i+BATCH_SIZE]
                batch_y = y[i:i+BATCH_SIZE]
                hidden = net.init_hidden(batch_X)
                optimiser.zero_grad()
                outputs, _ = net(batch_X, hidden)
                loss = loss_function(outputs, batch_y)
                loss_sum = loss_sum + loss_sum_function(outputs, batch_y)/rnn_seq_length # divide by 10 to account for len(input)
                loss.backward()
                optimiser.step()
    else:
        net_type = 'invalid'
    sys.stderr = sys.__stderr__
    if net_type == 'invalid':
        print("Invalid neural network type. Ensure last argument is 'fnn' or 'rnn'")
    #print("Loss:",loss,", Sum",loss_sum)
    # For the RNN len(inputs) is the number of training data sets, therefore, divide by 10 (the seq_length). As done for loss_sum above
    return loss_sum/(len(inputs)*EPOCHS) # Return training MSE error