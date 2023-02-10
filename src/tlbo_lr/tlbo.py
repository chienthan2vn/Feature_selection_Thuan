import numpy as np
from numpy.random import rand
import sys
sys.path.append("./src")
from cso_lr.fun import fun
import math


def init_position(lb, ub, N, dim):
    X = np.zeros([N, dim], dtype='float')
    for i in range(N):
        for d in range(dim):
            X[i,d] = lb[0,d] + (ub[0,d] - lb[0,d]) * rand()        
    
    return X


def binary_conversion(X, thres, N, dim):
    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            if X[i,d] > thres:
                Xbin[i,d] = 1
            else:
                Xbin[i,d] = 0
    
    return Xbin


def boundary(x, lb, ub):
    if x < lb:
        x = lb
    if x > ub:
        x = ub
    
    return x

def jfs(xtrain, ytrain, opts):
    # Parameter
    ub = 1
    lb = 0
    thres = 0.5
    r = rand(1) #0 to 1

    N           = opts['N']
    max_iter    = opts['max_iter']
    # if 'Tf' in opts:
    #     Tf = opts['Tf']
    if 'r' in opts:
        r = opts['r']

    # Dimension
    dim = np.size(xtrain, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')

    # Initialize position
    X = init_position(lb, ub, N, dim)

    # Binary conversion
    Xbin  = binary_conversion(X, thres, N, dim)

    # Fitness at first iteration (Ham muc tieu)
    fit   = np.zeros([N, 1], dtype='float')
    Xgb   = np.zeros([1, dim], dtype='float')
    fitG  = float('inf')

    for i in range(N):
        fit[i,0] = fun(xtrain, ytrain, Xbin[i,:], opts)
        if fit[i,0] < fitG:
            Xgb[0,:] = X[i,:]
            fitG     = fit[i,0]

    # Pre
    curve = np.zeros([1, max_iter], dtype='float') 
    t     = 0

    curve[0,t] = fitG.copy()
    print("Generation:", t + 1)
    print("Best (FA):", curve[0,t])
    t += 1

    while t < max_iter:
        for i in range(N):
            Xmean = np.sum(X, axis=0)/np.size(X, axis=0)
            Tf = round(1+rand(1))
            X = X + r*(Xgb - Tf*Xmean)

            temp = np.zeros([1, dim], dtype='float')
            temp[0,:] = X[i,:]  
            Xbin = binary_conversion(temp, thres, 1, dim)

            # fitness
            fit[i,0]  = fun(xtrain, ytrain, Xbin[0,:], opts)
            
            # best update        
            if fit[i,0] < fitG:
                Xgb[0,:] = X[i,:]
                fitG     = fit[i,0]
                
        # Store result
        curve[0,t] = fitG.copy()
        print("Generation:", t + 1)
        print("Best (FA):", curve[0,t])
        t += 1

    # Best feature subset
    Gbin       = binary_conversion(Xgb, thres, 1, dim) 
    Gbin       = Gbin.reshape(dim)
    pos        = np.asarray(range(0, dim))    
    sel_index  = pos[Gbin == 1]
    num_feat   = len(sel_index)
    # Create dictionary
    tlbo_data = {'sf': sel_index, 'c': curve, 'nf': num_feat}
    
    return tlbo_data    