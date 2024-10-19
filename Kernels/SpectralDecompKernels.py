import numpy as np
import pywt

from Utils.Constants import LocalDataConstants, SpectralConstants
from Utils import KernelUtils as KU

def WavletSpectralDecomposer(data: np.ndarray, Band = 'All', wavelet = 'morl', return_freqs = False, Fs = LocalDataConstants.DefaulValues['Fs'], **kwargs):

    if str(data.shape[-1]) in SpectralConstants.WaveletParams['time_lims'].keys():

        TimeLims = SpectralConstants.WaveletParams['time_lims'][str(data.shape[-1])]

    else:

        TimeLims = [0, data.shape[-1] / Fs]

    if 'Spectral_Res' in kwargs.keys():

        SpecRes = kwargs['Spectral_Res']

    else:

        SpecRes = SpectralConstants.WaveletParams['Spectral_Res']

    if type(Band) == str:

        WiPa = SpectralConstants.WaveletParams['widths_param'][wavelet][str(Fs)][Band]

    else:

        WiPa = KU.AssignWidthsParams(BandRanges = Band, Fs = Fs, Spectral_Res = SpecRes)
    
    options = {

        'widths_param': WiPa,
        'time_lims': TimeLims,
        'Spectral_Res': SpecRes

    }

    options.update(kwargs)

    widths_param = options['widths_param']
    widths = np.geomspace(widths_param[0], widths_param[1], num = options['Spectral_Res'])

    time = np.linspace(options['time_lims'][0], options['time_lims'][1], data.shape[-1])

    assert data.ndim < 4 and data.ndim > 0, "Invalid data shape"

    CWTMat_Conn = []

    for _ in range(3 - data.ndim):

        data = np.reshape(data, (1,) + data.shape)

    [Trials, Channels, Length] = data.shape

    for Channel_Num in range(Channels):

        CWTMat = np.zeros((len(widths), Length))

        for i in range(Trials):

            cwtmatr, freqs = pywt.cwt(data[i, Channel_Num, :], widths, wavelet, sampling_period = np.diff(time).mean())
            
            CWTMat = CWTMat + 1 / Trials * cwtmatr

        CWTMat_Conn.append(CWTMat)

    outputs = []
    outputs.append(np.squeeze(np.array(CWTMat_Conn)))

    if return_freqs:

        outputs.append(freqs)

    return tuple(output for output in outputs)

def RidRihaczek(data: np.ndarray, Band = 'Alpha'):
    '''
    This is python implementation of rid_rihaczek4 function
    which was implemented in MATLAB by Munia in this repository
    https://github.com/muntam/TF-PAC

    The repository was implemented for
    Munia, T.T.K., Aviyente, S. Time-Frequency Based Phase-Amplitude
    Coupling Measure For Neuronal Oscillations. Sci Rep 9, 12441 (2019).
    https://doi.org/10.1038/s41598-019-48870-2

    This function computes reduced interference Rihaczek distribution

    Parameter:
        x: signal
        fbins=required frequency bins

    Returns:
        tfd = Generated reduced interference Rihaczek distribution

    Written by: Mahdi Kiani, March 2021
    '''

    x = data
    fbins = LocalDataConstants.DefaulValues['Fs']

    tbins = x.shape[0]
    amb = np.zeros((tbins, tbins))
    for tau in range(tbins):
        amb[tau, :] = (np.conj(x) * np.concatenate((x[tau:], x[:tau])))

    ambTemp = np.concatenate(
        (amb[:, KU.RoundUp(tbins / 2):], amb[:, : KU.RoundUp(tbins / 2)]), axis=1)
    amb1 = np.concatenate(
        (ambTemp[KU.RoundUp(tbins / 2):, :], ambTemp[: KU.RoundUp(tbins / 2), :]), axis=0)

    D = np.outer(np.linspace(-1, 1, tbins), np.linspace(-1, 1, tbins), )
    K = KU.ChWiKernel(D = D, L = D, A = 0.01)
    df = K[: amb1.shape[0], : amb1.shape[1]]
    ambf = amb1 * df

    A = np.zeros((fbins, tbins))
    tbins = tbins - 1

    if tbins != fbins:
        for tt in range(tbins):
            A[:, tt] = KU.DataWrap(ambf[:, tt], fbins)
    else:
        A = ambf

    tfd = np.fft.fft(A, axis=0)

    BandLims = SpectralConstants.BandsBounds[Band]

    BandTFD = tfd[KU.RoundUp(BandLims[0]) : KU.RoundUp(BandLims[1]) + 1, :]

    outputs = []

    outputs.append(BandTFD)

    return tuple(output for output in outputs)