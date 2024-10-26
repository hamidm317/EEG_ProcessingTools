import numpy as np
import scipy.signal as signal

from Utils import OrderedConnMeasures as OCM
from npeet import entropy_estimators as ee
from Utils.Constants import SpectralConstants as SC
from Utils.Constants import LocalDataConstants as LDC

from Kernels.SpectralDecompKernels import WavletSpectralDecomposer as WSD
from Kernels import SpectralDecompKernels
from Kernels import CorrKernels
from Modules import GRUniPy as GRU

def LRB_GC(x, y, specs, eta = 0.000001):

    # y -> Source Signal
    # x -> Target Signal    

    i = specs['i']
    j = specs['j']
    est_order = int(specs['est_orders'][i, j])

    uve = OCM.LRB_univar_e(y, est_order)
    mve = OCM.LRB_mulvar_e(y, x, est_order)

    return np.log(uve / (mve + eta))

def PLI(x, y, specs): # the PLI kernel could handle Band itself, but it is recommended not to use it, 
    # give the band-filtered data to the kernel.

    if 'Band' in specs.keys():

        Band = specs['Band']

    else:

        Band = 'All'

    if Band == 'All':
    
        x_b = x
        y_b = y

        x_a = signal.hilbert(x_b)
        y_a = signal.hilbert(y_b)

    else:

        x_a = WSD(x, Band = Band, wavelet = 'cmor')[0]
        y_a = WSD(y, Band = Band, wavelet = 'cmor')[0]
    
    phase_signs = np.sign(np.angle(x_a) - np.angle(y_a))
    
    return np.abs(np.mean(phase_signs))

def PLV(x, y, specs): # the PLI kernel could handle Band itself, but it is recommended not to use it, 
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
        y_b = WSD(y, Band = Band)[0]

    x_a = signal.hilbert(x_b)
    y_a = signal.hilbert(y_b)
    
    phase_exps = np.exp(1j * (np.angle(x_a) - np.angle(y_a)))
    
    return np.abs(np.mean(phase_exps))

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
    Y_lagged = OCM.roll_mat_gen(y, est_order, end_include = False)
    X_lagged = OCM.roll_mat_gen(x, est_order, end_include = False)
    
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
        y_b = WSD(y, Band = Band)[0]

    x_a = signal.hilbert(x_b)
    y_a = signal.hilbert(y_b)
    
    phase_HSs = np.heaviside(np.angle(x_a) - np.angle(y_a), 0.5)
    
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
        y_b = WSD(y, Band = Band)[0]

    x_a = signal.hilbert(x_b)
    y_a = signal.hilbert(y_b)
    
    numerator = np.abs(np.mean(np.angle(x_a) - np.angle(y_a)))
    denominator = np.mean(np.abs(np.angle(x_a) - np.angle(y_a)))
    
    return np.mean(numerator / (denominator + eta))

def PAC(x, y, specs):

    # x -> Consider X as Amplitude Signal
    # y -> Consider Y as Phase Signal

    FInKer = specs['FilterInKernel']

    x = x / np.sqrt(np.var(x))
    y = y / np.sqrt(np.var(y))

    if FInKer:

        AmpBand = specs['AmpBand']
        PhaseBand = specs['PhaseBand']

        SpecDecompMethod = specs['SpecDecompKernel']

        SpecDecomp_Kernel = getattr(SpectralDecompKernels, LDC.names['LocalCM'][SpecDecompMethod])

        X_Filtered = SpecDecomp_Kernel(data = x, Band = AmpBand)
        Y_Filtered = SpecDecomp_Kernel(data = y, Band = PhaseBand)

        AmpSignal = GRU.PowerPhaseExt(np.mean(X_Filtered[0], axis = 0), return_value = 'Power')[0]
        PhaSignal = GRU.PowerPhaseExt(np.mean(Y_Filtered[0], axis = 0), return_value = 'Phase')[0]

    else:

        AmpSignal = GRU.PowerPhaseExt(x, return_value = 'Power')[0]
        PhaSignal = GRU.PowerPhaseExt(y, return_value = 'Phase')[0]

    PhaAmpCorrMethod = specs['PhaseAmplitudeCorrelateCalc']

    PACorr_Kernel = getattr(CorrKernels, PhaAmpCorrMethod)
    PAC_Value = PACorr_Kernel(AmpSig = AmpSignal, PhaSig = PhaSignal)

    return PAC_Value

def PCor(x, y, specs):

    if 'Domain' in specs.keys():

        Domain = specs['Domain']

    else:

        Domain = 'P2P'

    if 'Band' in specs.keys():

        Band = specs['Band']

    else:

        Band = 'All'

    if Band == 'All':
    
        x_b = x
        y_b = y

    else:

        x_b = np.mean(WSD(x, Band = Band)[0], axis = 0)
        y_b = np.mean(WSD(y, Band = Band)[0], axis = 0)

    if Domain == 'P2P':

        x_a = np.abs(signal.hilbert(x_b))
        y_a = np.abs(signal.hilbert(y_b))

    else:

        x_a = x_b
        y_a = y_b
        
    CorrCoef = np.corrcoef(x_a, y_a, rowvar = True)
    
    return CorrCoef[0, 1]

def DCG_PAC(x, y, specs):

    AmpBand = specs['AmpBand']
    PhaseBand = specs['PhaseBand']

    SpecDecompMethod = specs['SpecDecompKernel']

    AmpSpecRes = specs['AmpBins']
    PhaseSpecRes = specs['PhaseBins']

    SpecDecomp_Kernel = getattr(SpectralDecompKernels, LDC.names['LocalCM'][SpecDecompMethod])

    X_Filtered = SpecDecomp_Kernel(data = x, Band = AmpBand, Spectral_Res = AmpSpecRes)
    Y_Filtered = SpecDecomp_Kernel(data = y, Band = PhaseBand, Spectral_Res = PhaseSpecRes)

    AmpSignal = GRU.PowerPhaseExt(X_Filtered[0], return_value = 'Power')[0]
    PhaSignal = GRU.PowerPhaseExt(Y_Filtered[0], return_value = 'Phase')[0]

    PhaAmpCorrMethod = specs['PhaseAmplitudeCorrelateCalc']
    PACorr_Kernel = getattr(CorrKernels, PhaAmpCorrMethod)

    CDG_Mat = np.zeros((AmpSpecRes, PhaseSpecRes))

    for Ai, NB_AmpSignal in enumerate(AmpSignal):

        for Pi, NB_PhaSignal in enumerate(PhaSignal):

            CDG_Mat[Ai, Pi] = PACorr_Kernel(AmpSig = NB_AmpSignal, PhaSig = NB_PhaSignal)

    return CDG_Mat