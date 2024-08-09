import numpy as np
from Utils.Constants import DC_Constants
from Kernels import ConKernels

def BIC_calc(MeanSqErr, n, k):
    
    # BIC = k * ln(n) + n * ln(mean sum of residuals)
    
    return k * np.log(n) + n * np.log(MeanSqErr)

def AIC_calc(MeanSqErr, n, k):
    
    # AIC = 2k + n * ln(mean sum of residuals)
    
    return 2 * k + n * np.log(MeanSqErr)

def AssignOrderEstFunction(kernelName):

    return getattr(OrderKernels, kernelName)

def AssignConnectivityFunction(KernelName):

    CoreKernelFunction = getattr(ConKernels, KernelName)

    KernelProperties = DC_Constants.Properties[KernelName]

    return CoreKernelFunction, KernelProperties