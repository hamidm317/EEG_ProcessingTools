import numpy as np

from Utils import KernelUtils as KU

def KullbackLeibler(Distributions, base = 10):

    assert len(Distributions) == 2, "You should pass distributions of TWO r.vs"

    P = Distributions[0]
    Q = Distributions[1]

    assert len(P) == len(Q), "The same number of realization must be used"

    '''
    KullbackLeibler(P, Q) == D(P || Q) := \sigma(P(x) * log(P(x) / Q(x))) over x of space

    '''

    P = P / np.sum(P)
    Q = Q / np.sum(Q)

    KLD = np.sum([KU.KLKern(P[x], Q[x]) / np.log(base) for x in range(len(P))])

    return KLD

def ShannonEntropy(Distribution, base = 2):

    P = Distribution[0]

    ShaEnt = -1 * np.sum([KU.WeightedInfo(p) / np.log(base) for p in P])

    return ShaEnt