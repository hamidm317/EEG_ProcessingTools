import numpy as np
import pywt

from Utils.OrderedConnMeasures import roll_mat_gen


def WaveLetFrequencyCalibration(widths: np.ndarray, time: np.ndarray, wavelet = 'morl'):

    data = np.random.normal(loc = 0, scale = 1, size = (len(time)))

    cwtmatr, freqs = pywt.cwt(data, widths, wavelet, sampling_period = np.diff(time).mean())

    return freqs

def RoundUp(n):
    
    return int(np.floor(n + 0.5))

def ChWiKernel(D: np.ndarray, L: np.ndarray, A: int = None):
    '''

    CHWI_KRN Choi-Williams kernel function.

    https://en.wikipedia.org/wiki/Bilinear_time%E2%80%93frequency_distribution#Choi%E2%80%93Williams_distribution_function

    K = _chwi_krn(D, L, A) returns the values K of the Choi-Williams kernel function
    evaluated at the doppler-values in matrix D and the lag-values in matrix L.
    Matrices D and L must have the same size. The values in D should be in the range
    between -1 and +1 (with +1 being the Nyquist frequency). The parameter A is
    optional and controls the "diagonal bandwidth" of the kernel. Matrix K is of the
    same size as the matrices D and L. Parameter A defaults to 10 if omitted.

    Copyright (c) 1998 by Robert M. Nickel
    Revision: 1.1.1.1
    Date: 2001/03/05 09:09:36

    Written by: Mahdi Kiani, March 2021

    '''

    if A is None:
        A = 10
    K = np.exp((-1/(A*A)) * (D*D*L*L))

    return K

def DataWrap(x: np.ndarray, n: int) -> np.ndarray:
    '''
    The calculation of signal spectrum, such as periodogram, uses FFT internally, 
    where the length of FFT is denoted as NFFT. In theory, when using FFT, 
    the signal in both time domain and frequency domain are discrete and periodic, 
    where the period is given by NFFT. Hence, if you specify an NFFT that is less 
    than the signal length, it actually introduces the aliasing in the time domain 
    and make the signal (even if its length is N>NFFT) periodic with NFFT. 
    When you take FFT of this sequence, you are working with this aliased sequence. 
    This is what datawrap do for you. 

    For example: Sequence 1 2 3 4 5, period 5, it returns
        1 2 3 4 5
                  1 2 3 4 5
                            1 2 3 4 5
        --------------------------------
              ... 1 2 3 4 5 ...

    i.e., original series. assume a period of 3, then it looks like

        1 2 3 4 5
              1 2 3 4 5
                    1 2 3 4 5
        ------------------------
          ... 5 7 3 ...

    A sequence that is wrapped around and has only a length of 3.

    >>> _datawrap(range(1, 6),3)
    array([5, 7, 3])

    '''
    return np.array([sum(x[i::n]) for i in range(n)])

def ProxyDistribution(Mat, Bins):

    assert len(Mat) == 2, "Mat must include two arrays of Random Variables"

    MastersRV = Mat[0]
    SlavesRV = Mat[1]

    SlavesLocations = []

    for Slave in SlavesRV:

        SlavesLocations.append(BinLocation(Slave, Bins))

    LocationsWealth = []

    for Location in range(len(Bins)):

        LocationsWealth.append(np.sum(MastersRV[np.where([SlaveLocation == Location for SlaveLocation in SlavesLocations])]))

    LocationsWealth = np.array(LocationsWealth)

    return LocationsWealth / np.sum(LocationsWealth)

def BinLocation(x, BinEdges):

    if x > np.max(BinEdges):

        return [len(BinEdges) - 1]
    
    elif x < np.min(BinEdges):

        return [0]
    
    else:

        ActualBins = roll_mat_gen(BinEdges, k = 2)

        BinExistence = np.where([x < ActualBin[1] and x >= ActualBin[0] for ActualBin in ActualBins])

        return np.squeeze(BinExistence[0][0])
    
def KLKern(P, Q):

    if P == 0:

        return 0
    
    if Q == 0:

        return 100000000
    
    return P * np.log(P / Q)

def WeightedInfo(p):

    if p == 0:

        return 0
    
    else:

        return p * np.log(p)