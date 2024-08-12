import numpy as np
import scipy.signal as signal

import Utils.GC as GC
import Utils.TE as TE_U
from npeet import entropy_estimators as ee

def LRB_GC(x, y, specs):

    # y -> Source Signal
    # x -> Target Signal    

    i = specs['i']
    j = specs['j']
    est_order = int(specs['est_orders'][i, j])

    uve = GC.LRB_univar_e(x, est_order)
    mve = GC.LRB_mulvar_e(x, y, est_order)

    return np.log(uve / mve)

def PLI(x, y, specs):
    
    x_a = signal.hilbert(x)
    y_a = signal.hilbert(y)
    
    phase_signs = np.sign(np.angle(x_a) - np.angle(y_a))
    
    return np.abs(np.mean(phase_signs))

def PLI_Comp(x, y, specs):
    
    phase_signs = np.sign(np.angle(x) - np.angle(y))
    
    return np.abs(np.mean(phase_signs))

def PIB_GC(x, y, specs):

    # it is much better to give access to orders to users!
    GC_val = []
    univar_error = []
    mulvar_error = []

    i = specs['i']
    j = specs['j']
    order = int(specs['est_orders'][i, j])    

    a_est_uni = GC.univar_AR_est(x, order)

    a_order = order
    b_order = a_order

    a_est_mul, b_est_mul = GC.mulvar_AR_est(x, y, a_order, b_order)

    univar_error.append(GC.a_estimation_err(a_est_uni, x))
    mulvar_error.append(GC.ab_estimation_err(a_est_mul, b_est_mul, x, y))
        
    GC_val.append(np.log(univar_error[-1] / mulvar_error[-1]))
    
    return GC_val, univar_error, mulvar_error

def TE(x, y, specs):

    # x -> Source Signal
    # y -> Target Signal

    w_x = specs['w_x']
    w_y = specs['w_y']

    d_x = specs['d_x']
    d_y = specs['d_y']
    
    X_lagged = TE_U.generate_lagged_vectors(x, w_x)
    Y_lagged = TE_U.generate_lagged_vectors(y, w_y)

    X_lagged = X_lagged[:-d_x-1]
    Y_lagged = Y_lagged[:-d_y-1]

    max_index = max(w_x + d_x , w_y + d_y)
    if w_x + d_x == max_index:
        Y_lagged = Y_lagged[max_index- w_y - d_y:]
    else:         
        X_lagged = X_lagged[max_index- w_x - d_x:]
    

    Y_t = y[max_index:]
    
    return ee.cmi(Y_t, X_lagged, Y_lagged)