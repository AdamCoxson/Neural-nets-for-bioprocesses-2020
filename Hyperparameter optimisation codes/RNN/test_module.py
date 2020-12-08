import torch
from torch.autograd import Variable
import numpy as np

def test(test_inputs, test_labels, net):
    seq_length = 10 # Number of testable datapoints in each experimental trial (total length - 1)
    n_testing_sets = 5 # Number of expt. trials data sets used for testing data
    testing_set_mse_vals = [0]*n_testing_sets
    squared_error_X = []
    net.eval()
    test_X = Variable(torch.Tensor(test_inputs)) 
    test_y = Variable(torch.Tensor(test_labels))
    hidden = net.init_hidden(test_X)
    with torch.no_grad():
        net_out, _ = net(test_X, hidden)  # Hidden state not required for manual feeding


    for index1, element in enumerate(test_y):
        for index2, row in enumerate(element):
            X_error = row[0] - net_out[index1][index2][0]
            squared_error_X.append(float(X_error)**2)

    
    for i in range(0,n_testing_sets):
        testing_set_mse_vals[i] = sum(squared_error_X[0+(i*seq_length):seq_length+(i*seq_length)])/seq_length
    avg_mse = np.mean(np.array(testing_set_mse_vals))
    
    prediction = []
    bc_init, ga, co, li = [], [], [], [] # biomass concentration is the only non-constant
    for i in range(0, n_testing_sets):
        bc_init.append(test_inputs[i][0][0])
        ga.append(test_inputs[i][0][1])
        co.append(test_inputs[i][0][2])
        li.append(test_inputs[i][0][3])
        
    net.sequence_length = 1              # Feed 1 input at a time to the network (offline prediction) 
    for i in range(0,n_testing_sets):
        hidden = net.init_hidden(Variable(torch.Tensor([[[]]])))            # Initialise hidden state with a batch size of 1
        bc = bc_init[i]
        prediction.append([float(bc), float(ga[i]), float(co[i]), float(li[i])])
        for j in range(0, seq_length):
            # Feed inputs with a batch size of 1, sequence length of 1 and feature vector length of 5 to the network
            net_out, hidden = net(Variable(torch.Tensor([[[bc, ga[i], co[i], li[i]]]])), hidden)
            bc = bc + net_out[0][0][0]
            prediction.append([float(bc), float(ga[i]), float(co[i]), float(li[i])])



    return avg_mse, testing_set_mse_vals, prediction

