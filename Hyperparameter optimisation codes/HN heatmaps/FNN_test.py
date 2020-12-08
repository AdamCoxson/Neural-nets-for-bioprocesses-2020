import torch
import numpy as np
import pandas as pd

def test(test_inputs, test_labels, net):
    net.eval()
    test_X = torch.Tensor(test_inputs).view(-1, 4)
    test_y = torch.Tensor(test_labels)

    predictionNumpy = []
    with torch.no_grad():
        for i in range(0, len(test_X)):
            net_out = net(test_X[i].view(-1, 4))
            predictionNumpy.append(net_out[0].numpy())              # The output from the net is a tensor which contains only one element which is a list. The list contains the 3 output values. We only want the list, not the tensoor containing one element which is a list.

    experimental = []
    for data in test_y:
        experimental.append(data.numpy())

    squared_error_X = []
    for i in range(0, len(experimental)):
            X_error = experimental[i][0] - predictionNumpy[i][0]
            squared_error_X.append(X_error**2)

    AVG_MSE = sum(squared_error_X[:])/len(squared_error_X)

    # GA_1, GA_2 = test_inputs[0][1], test_inputs[12][1]
    # CO2_1, CO2_2 = test_inputs[0][2], test_inputs[12][2]
    # LI_1, LI_2 = test_inputs[0][3], test_inputs[12][3]
    # predictions_online = []
    # for index, value in enumerate(test_inputs):
    #     BC = value[0] + predictionNumpy[index][0]

    #     if index < 12:
    #         predictions_online.append([BC, GA_1, CO2_1, LI_1])

    #     if index >= 12:
    #         predictions_online.append([BC, GA_2, CO2_2, LI_2])

    # predictions_offline = []
    # BC1, BC2 = test_inputs[0][0], test_inputs[12][0]

    # for index, value in enumerate(test_inputs):
    #     if index < 12:
    #         net_out = net(torch.Tensor([BC1, GA_1, CO2_1, LI_1]))
    #         BC = BC1 + net_out[0]   
    #         predictions_offline.append([float(BC), float(GA_1), float(CO2_1), float(LI_1)])
    #         BC1 = BC
        
    #     if index >= 12:
    #         net_out = net(torch.Tensor([BC2, GA_2, CO2_2, LI_2]))
    #         BC = BC2 + net_out[0] 
    #         predictions_offline.append([float(BC), float(GA_2), float(CO2_2), float(LI_2)])
    #         BC2 = BC
            
    return AVG_MSE  #, predictions_online, predictions_offline
