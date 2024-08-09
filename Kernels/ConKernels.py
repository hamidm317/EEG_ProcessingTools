import numpy as np
import scipy.signal as signal

import Utils.GC as GC

def LRB_GC(x, y, specs):

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
        
    # print("Order is", order, "and Granger Causality is", np.log(a_estimation_err(a_est_uni, x_t) / ab_estimation_err(a_est_mul, b_est_mul, x_t, y_t)), "Univar Error is", a_estimation_err(a_est_uni, x_t), "and mulvar error is", ab_estimation_err(a_est_mul, b_est_mul, x_t, y_t))
        
    return GC_val, univar_error, mulvar_error