import numpy as np

from Utils import KernelUtils as KU
from Kernels import GeneralKernels as GK

from Utils.Constants import KernelConstants as KC

def MeanVectorLength(AmpSig, PhaSig):

    assert len(PhaSig) == len(AmpSig), "Amplitude and Phase signals must be same length"

    PhiTerm = np.exp(1j * PhaSig)
    WTN = AmpSig * (PhiTerm)

    MVL = np.abs(np.mean(WTN))

    return MVL

def ModulationIndex(AmpSig, PhaSig, NumberOfBins = None, DistanceFunction = 'KullbackLeibler', HypoTestLoopLength = None):

    assert len(AmpSig) == len(PhaSig), "Signal must have same lengths"

    HypoTestLoopLength = KC.DistanceKernels[DistanceFunction]['HypoTestLoopLength']

    GenerateNull = not KC.DistanceKernels[DistanceFunction]['Deterministic']

    if NumberOfBins == None:

        NumberOfBins = 36

    AmpProxyDist = KU.ProxyDistribution([AmpSig, PhaSig], Bins = np.linspace(-np.pi, np.pi, NumberOfBins))

    DistanceKernel = getattr(GK, DistanceFunction)

    DistanceToNulls = []

    for _ in range(HypoTestLoopLength):

        NullDistribution = np.ones(shape = (NumberOfBins,)) / NumberOfBins

        DistanceInput = [AmpProxyDist]

        if GenerateNull:
            
            DistanceInput.append(NullDistribution)

        DistanceToNulls.append(DistanceKernel(DistanceInput))

    MI = np.mean(DistanceToNulls)

    return MI