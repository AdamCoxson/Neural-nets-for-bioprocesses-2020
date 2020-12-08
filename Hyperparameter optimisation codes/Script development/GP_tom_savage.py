# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 17:14:53 2020

@author: adamc
"""

import numpy as np 
import GPy
import numpy.random as rnd 
from scipy.optimize import minimize


def LHS(bounds,p):
    '''
    Performs a Latin Hypercube sample of size p within bounds
    '''
    d = len(bounds)
    sample = np.zeros((p,len(bounds)))
    for i in range(0,d):
        sample[:,i] = np.linspace(bounds[i,0],bounds[i,1],p)
        rnd.shuffle(sample[:,i])
    return sample 

def styblinskitang(X):
    '''
    INPUTS
    X: arguments of the Styblinski-Tang Function

    OUTPUTS
    f : evaluation of the Styblinski-Tang function given the inputs
    
    DOMAIN         : [-5,5]
    DIMENSIONS     : any
    GLOBAL MINIMUM : f(x)=(-39.166*d) x=[-2.9035,...,-2.9035] 
    '''
    f_sum=sum((X[i]**4)-(16*(X[i]**2))+(5*X[i]) for i in range(len(X)))
    return f_sum/2

dim = 2
bounds = np.array([[-5,5] for i in range(dim)])
f = styblinskitang

# gaining initial sample
sample_size = 10
sample = LHS(bounds,sample_size)
overall_func_count = sample_size 

# evaluating function values for sample
f_values = np.zeros((sample_size,1))
for i in range(sample_size):
    f_values[i,:] = f(sample[i,:])

f_values_unnorm = np.copy(f_values)

# normalising for use in GP
f_values = (f_values-np.mean(f_values))/np.std(f_values)

# defining GP kernel and model 
kernel = GPy.kern.RBF(input_dim = dim, ARD = True,variance=1,lengthscale=1)
m = GPy.models.GPRegression(sample,f_values,kernel)

# optimising GP (can change optimisation runs if needed)
m.optimize(messages=False)
m.optimize_restarts(0)

# cost function (can change variance term)
def GP_eval(x,m):
    mean, var = m.predict(np.array([x]))
    return mean.item() + var.item() 

# optimisation of GP, multi-start gradient based 
def GP_optimize(m,bounds):
    restarts = 10 # number of optimisation restarts
    initial_sols = LHS(bounds,restarts)
    f_min = 100000
    for i in range(restarts):
        sol = minimize(GP_eval,x0=initial_sols[i,:],args=(m),bounds=bounds)
        if sol.fun < f_min:
            best_x = sol.x 
            f_min = sol.fun 
    return best_x 

# GP optimisation iterations 
overall_iterations = 20

for i in range(overall_iterations):
    x_new = GP_optimize(m,bounds)
    f_new = f(x_new)
    overall_func_count += 1 
    print('func value: ',np.round(f_new,3),' overall func evaluations: ',overall_func_count)

    sample = np.append(sample,[x_new],axis=0)
    f_values_unnorm = np.append(f_values_unnorm,[[f_new]],axis=0)

    f_values = (f_values_unnorm-np.mean(f_values_unnorm))/np.std(f_values_unnorm)

    m = GPy.models.GPRegression(sample,f_values,kernel)
    m.optimize(messages=False)
    m.optimize_restarts(0)

best_sol_index = np.argmin(f_values_unnorm)
best_sol = sample[best_sol_index,:]

print('final solution at: ',best_sol)