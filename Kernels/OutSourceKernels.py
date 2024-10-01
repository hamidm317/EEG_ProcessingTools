import numpy as np

from Utils.Constants import LocalDataConstants as LDC
from Kernels import SpectralDecompKernels
from Kernels import ConKernels as CK

def OS_PAC(Data, window_length = 100, overlap_ratio = 0.98, **kwargs):

    options = {

        'inc_channels': np.arange(Data.shape[-2]),
        'orders_matrix': None,
        'PhaseBand': 'Theta',
        'AmpBand': 'Gamma',
        'SpecDecompKernel': 'Wavelet',
        'PhaseAmplitudeCorrelateCalc': 'MeanVectorLength',
        'PermuteBro': False

    }

    options.update(kwargs)

    assert Data.ndim == 2 or Data.ndim == 3, "Your Data must be 3 or 2 Dimensional, (Trials (optional), Channels, Time)"

    if Data.ndim == 2:

            Data = np.reshape(Data, (1, Data.shape[0], Data.shape[1]))

    time_length = Data.shape[-1]
    number_of_windows = int((time_length - window_length) / ((1 - overlap_ratio) * window_length)) + 1
    number_of_trials = Data.shape[0]

    channels = options['inc_channels']

    if len(channels) > 1 and np.all([type(electrode) in [int, np.int32] for electrode in channels]):

            i_channels = channels
            j_channels = channels

    elif len(channels) == 2:

        i_channels = channels[0]
        j_channels = channels[1]

    else:

        assert False, "Invalid Channels matrix shape"

    specs = {
         
        'FilterInKernel': False,
        'PhaseAmplitudeCorrelateCalc': options['PhaseAmplitudeCorrelateCalc']
    }

    SpecDecompMethod = options['SpecDecompKernel']
    SpecDecomp_Kernel = getattr(SpectralDecompKernels, LDC.names['LocalCM'][SpecDecompMethod])

    AmplitudeData = SpecDecomp_Kernel(Data, Band = options['AmpBand'])[0]
    PhaseData = SpecDecomp_Kernel(Data, Band = options['PhaseBand'])[0]

    DC_values = np.zeros((number_of_trials, number_of_windows, len(i_channels), len(j_channels)))

    for trial_i in range(number_of_trials):

        for win_step in range(number_of_windows):

            for i, channel_a in enumerate(i_channels):

                for j, channel_b in enumerate(j_channels):

                    win_stp = int((win_step) * (1 - overlap_ratio) * window_length)
                    win_enp = win_stp + window_length

                    if options['PermuteBro']:

                        x_t = np.random.permutation(AmplitudeData[trial_i, channel_a, win_stp : win_enp])
                        y_t = np.random.permutation(PhaseData[trial_i, channel_b, win_stp : win_enp])

                    else:

                        x_t = AmplitudeData[trial_i, channel_a, win_stp : win_enp]
                        y_t = PhaseData[trial_i, channel_b, win_stp : win_enp]

                    win_DC_val = CK.PAC(x_t, y_t, specs)

                    DC_values[trial_i, win_step, i, j] = win_DC_val

    return DC_values