import numpy as np
import scipy.signal as signal

from Utils import OrderedConnMeasures as OCM
from npeet import entropy_estimators as ee
from Utils.Constants import SpectralConstants as SC

from Utils.SpectralDeco import WavletSpectralDecomposer as WSD

def LRB_GC(x, y, specs):

    # y -> Source Signal
    # x -> Target Signal    

    i = specs['i']
    j = specs['j']
    est_order = int(specs['est_orders'][i, j])

    uve = OCM.LRB_univar_e(y, est_order)
    mve = OCM.LRB_mulvar_e(y, x, est_order)

    return np.log(uve / (mve + 0.000001))

def PLI(x, y, specs): # the PLI kernel could handle Band itself, but it is recommended not to use it, 
    # give the band-filtered data to the kernel.

    if 'Band' in specs.keys():

        Band = specs['Band']

    else:

        Band = 'All'

    if Band == 'All':
    
        x_b = x
        y_b = y

    else:

        x_b = WSD(x, Band = Band)[0]
        y_b = WSD(x, Band = Band)[0]

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

    a_est_uni = OCM.univar_AR_est(x, order)

    a_order = order
    b_order = a_order

    a_est_mul, b_est_mul = OCM.mulvar_AR_est(x, y, a_order, b_order)

    univar_error.append(OCM.a_estimation_err(a_est_uni, x))
    mulvar_error.append(OCM.ab_estimation_err(a_est_mul, b_est_mul, x, y))
        
    GC_val.append(np.log(univar_error[-1] / mulvar_error[-1]))
    
    return GC_val, univar_error, mulvar_error

def TE(x, y, specs):

    i = specs['i']
    j = specs['j']
    est_order = int(specs['est_orders'][i, j])

    Y_t = y[est_order:]
    Y_lagged = OCM.roll_mat_gen(y, est_order)
    X_lagged = OCM.roll_mat_gen(x, est_order)
    
    return ee.cmi(Y_t, X_lagged, Y_lagged)

def dPLI(x, y, specs):

    if 'Band' in specs.keys():

        Band = specs['Band']

    else:

        Band = 'All'

    if Band == 'All':
    
        x_b = x
        y_b = y

    else:

        x_b = WSD(x, Band = Band)[0]
        y_b = WSD(x, Band = Band)[0]

    x_a = signal.hilbert(x)
    y_a = signal.hilbert(y)
    
    phase_HSs = np.heaviside(np.angle(x_a) - np.angle(y_a))
    
    return np.mean(phase_HSs)

def wPLI(x, y, specs, eta = 0.00000001):

    if 'Band' in specs.keys():

        Band = specs['Band']

    else:

        Band = 'All'

    if Band == 'All':
    
        x_b = x
        y_b = y

    else:

        x_b = WSD(x, Band = Band)[0]
        y_b = WSD(x, Band = Band)[0]

    x_a = signal.hilbert(x)
    y_a = signal.hilbert(y)
    
    numerator = np.abs(np.mean(np.angle(x_a) - np.angle(y_a)))
    denominator = np.mean(np.abs(np.angle(x_a) - np.angle(y_a)))
    
    return numerator / (denominator)