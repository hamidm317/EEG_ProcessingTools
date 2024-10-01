import numpy as np
import scipy.stats as sps

def ExtNormalParameters(Population):

    return np.array([np.mean(Population, axis = 0), np.var(Population, axis = 0)])

def NormalPDF(x, Params):

    loc = Params[0]
    scale = np.sqrt(Params[1])

    p = 1 / (scale * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - loc) / scale) ** 2)

    return p

def MatWhiU(Sample, Population):

    return sps.mannwhitneyu(x = Population, y = Sample).pvalue