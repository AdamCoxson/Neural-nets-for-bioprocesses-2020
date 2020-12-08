import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import sys

def train(net, inputs, labels, EPOCHS, l_rate, BATCH_SIZE):
    # create a text trap and redirect stdout
    sys.stderr = open(os.devnull, "w")
    net.train()
    optimiser = optim.Adam(net.parameters(), lr = l_rate) 
    # net.parameters():  all of the adjustable parameters in our network. 
    # lr: a hyperparameter adjusts the size of the step that the optimizer will take to minimise the loss.
    loss_function = nn.MSELoss(reduction='mean')
    X = torch.Tensor(inputs).view(-1, 4)
    y = torch.Tensor(labels)
    
    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(X), BATCH_SIZE)):
            batch_X = X[i:i+BATCH_SIZE].view(-1, 4)
            batch_y = y[i:i+BATCH_SIZE]
            optimiser.zero_grad()
            outputs = net(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimiser.step()
    sys.stderr = sys.__stderr__