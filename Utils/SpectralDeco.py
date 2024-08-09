import numpy as np
import pywt

from Utils.Constants import LocalDataConstants, SpectralConstants

def WavletSpectralDecomposer(widths: np.ndarray, data, inc_channels: list, time: np.ndarray, wavelet = 'morl', band_range_output = True):

    assert data.ndim < 4 and data.ndim > 0, "Invalid data shape"
    assert data.shape[-1] == len(time), "Time and data lengths must be same"

    if data.ndim > 1:
        
        assert data.shape[1] >= len(inc_channels), "Included channels must be less than/equal to data channels"

    CWTMat_Conn = []

    if data.ndim == 3:

        [Trials, Channels, Length] = data.shape


        for Channel_Num in inc_channels:

            CWTMat = np.zeros((len(widths), Length))

            for i in range(Trials):

                cwtmatr, freqs = pywt.cwt(data[i, Channel_Num, :], widths, wavelet, sampling_period = np.diff(time).mean())
                
                CWTMat = CWTMat + 1 / Trials * cwtmatr

            CWTMat_Conn.append(CWTMat)

        
    elif data.ndim == 2:

        [Channels, Length] = data.shape

        for Channel_Num in inc_channels:

            CWTMat, freqs = pywt.cwt(data[Channel_Num, :], widths, wavelet, sampling_period = np.diff(time).mean())

            CWTMat_Conn.append(CWTMat)

    else:

        Length = len(data)

        CWTMat, freqs = pywt.cwt(data, widths, wavelet, sampling_period = np.diff(time).mean())

        CWTMat_Conn.append(CWTMat)


    
    band_range = []

    if band_range_output:

        band_range = [np.array([np.where(freqs > SpectralConstants.BandsBounds[band][0])[-1][-1], np.where(freqs < SpectralConstants.BandsBounds[band][1])[0][0]]) for band in LocalDataConstants.names['freq_bands']]
  
    return np.array(CWTMat_Conn), band_range, freqs

def FrequencyCalibration(widths: np.ndarray, time: np.ndarray, wavelet = 'morl'):

    data = np.random.normal(loc = 0, scale = 1, size = (len(time)))

    cwtmatr, freqs = pywt.cwt(data, widths, wavelet, sampling_period = np.diff(time).mean())

    return freqs