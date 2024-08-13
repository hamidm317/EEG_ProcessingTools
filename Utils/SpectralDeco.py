import numpy as np
import pywt

from Utils.Constants import LocalDataConstants, SpectralConstants

def WavletSpectralDecomposer(data: np.ndarray, Band = 'All', wavelet = 'morl', **kwargs): #wavelet = 'morl', band_range_output = True, widths: np.ndarray):

    options = {

        'widths_param': SpectralConstants.WaveletParams['widths_param'][wavelet][Band],
        'time_lims': SpectralConstants.WaveletParams['time_lims'][str(data.shape[-1])],
        'Spectral_Res': SpectralConstants.WaveletParams['Spectral_Res']

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
    outputs.append(freqs)

    return tuple(output for output in outputs)

def FrequencyCalibration(widths: np.ndarray, time: np.ndarray, wavelet = 'morl'):

    data = np.random.normal(loc = 0, scale = 1, size = (len(time)))

    cwtmatr, freqs = pywt.cwt(data, widths, wavelet, sampling_period = np.diff(time).mean())

    return freqs