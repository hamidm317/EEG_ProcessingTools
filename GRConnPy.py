import numpy as np
from Utils.General import *

from kneed import KneeLocator

def DynamicConnectivityMeasure(Data: np.ndarray, window_length = 100, overlap_ratio = 0.98, kernel = 'PLI', **kwargs):

    # Issues:
    # # It must be possible to choose two groups of channel in directed connectivities!
    # # How to handle spectral decomposed data?

    options = {

        'inc_channels': np.arange(Data.shape[-2]),
        'orders_matrix': None,
        'phase_freq': [4, 8],
        'amp_freq': [36, 42],
        'PAC_DecompMethod': 'wavelet',
        'd_x': 10,
        'd_y': 10,
        'w_x': 1,
        'w_y': 1

    }

    options.update(kwargs)

    assert len(options) == 9, "Invalid Keyword"

    CoreKernelFunction, KernelProperties = AssignConnectivityFunction(kernel)

    if KernelProperties['lagged']:

        print("You choose a lagged kernel")

        assert options['orders_matrix'] != None, "An orders matrix (lags) must be provided for lagged connectivities"

    else:

        print("You choose an instantaneous kernel")
    
    
    assert type(options['inc_channels']) == list or type(options['inc_channels']) == np.ndarray, "Included Channels must be a list or np array"
    assert len(options['inc_channels']) > 1 and len(options['inc_channels']) <= Data.shape[-2], "At least two channels and at most number of channels of data must be chosen"

    channels = options['inc_channels']
    order_EF = False

    if type(options['orders_matrix']) == int or type(options['orders_matrix']) == float:

        order_EF = True
        orders_mat = np.ones((len(channels), len(channels))) * options['orders_matrix']

    else:

        if type(options['orders_matrix']) == list or type(options['orders_matrix']) == np.ndarray:

            order_EF = True
            orders_mat = np.array(options['orders_matrix'])

            assert orders_mat.ndim == 2 and orders_mat.shape[0] == orders_mat.shape[1] and orders_mat.shape[0] == len(channels), "Improper Orders Matrix dimensions!"

    assert order_EF, "Invalid type of orders matrix"

    assert len(options['phase_freq']) == 2 and len(options['amp_freq']) == 2, "Phase and Amplitude Frequencies must be a list with length equal to 2"

    specs = {'est_orders': orders_mat}
    specs['amp_freq'] = options['amp_freq']
    specs['phase_freq'] = options['phase_freq']

    specs['d_x'] = options['d_x']
    specs['d_y'] = options['d_y']
    specs['w_x'] = options['w_x']
    specs['w_y'] = options['w_y']

    assert Data.ndim == 2 or Data.ndim == 3, "Your Data must be 3 or 2 Dimensional, (Trials (optional), Channels, Time)"

    if Data.ndim == 2:

        Data = np.reshape(Data, (1, Data.shape[0], Data.shape[1]))

    
    time_length = Data.shape[-1]
    number_of_channels = len(channels)
    number_of_windows = int((time_length - window_length) / ((1 - overlap_ratio) * window_length)) + 1
    number_of_trials = Data.shape[0]

    DC_values = np.zeros((number_of_trials, number_of_windows, number_of_channels, number_of_channels))

    for trial_i in range(number_of_trials):

        for win_step in range(number_of_windows):

            # print("In Progress", win_step / number_of_windows * 100, "% ...")
            # it is not beautiful brother!

            for i, channel_a in enumerate(channels):

                if KernelProperties['directed']:

                    for j, channel_b in enumerate(channels):

                        win_stp = int((win_step) * (1 - overlap_ratio) * window_length)
                        win_enp = win_stp + window_length

                        x_t = Data[trial_i, channel_a, win_stp : win_enp]
                        y_t = Data[trial_i, channel_b, win_stp : win_enp]

                        specs['i'] = i
                        specs['j'] = j

                        win_DC_val = CoreKernelFunction(x_t, y_t, specs)

                        DC_values[trial_i, win_step, i, j] = win_DC_val

                else:

                    for j, channel_b in enumerate(channels[:i]):

                        win_stp = int((win_step) * (1 - overlap_ratio) * window_length)
                        win_enp = win_stp + window_length

                        x_t = Data[trial_i, channel_a, win_stp : win_enp]
                        y_t = Data[trial_i, channel_b, win_stp : win_enp]

                        specs['i'] = i
                        specs['j'] = j

                        win_DC_val = CoreKernelFunction(x_t, y_t, specs)

                        DC_values[trial_i, win_step, i, j] = win_DC_val
                        DC_values[trial_i, win_step, j, i] = win_DC_val
            
    return np.squeeze(DC_values)

def OrderEstimate_byChannels(Data, max_order = 50, min_order = 2, leap_length = 2, **kwargs):

    # This channel estimate the order of AR process between channels,
    # Data -> m-by-n matrix, m is number of channels and n is length of data,
    # channels -> channels of Data matrix which orders must be calculated (list with maximum length m),
    # max_order -> maximum valid order to be included in estimation (less than n)
    # min_order -> minimum valid order to be included in estimation (more than zero)
    # leap_length -> orders likelihood will be computed with this leap_length

    options = {

        'inc_channels': np.arange(Data.shape[-2]),
        'kernel': 'LRB_GC'
    }

    options.update(kwargs)
    channels = options['inc_channels']

    assert len(channels) > 1 and len(channels) <= Data.shape[-2], "Included channels must be at least 2 and at most equal to the number of the data channels"
    
    number_of_channels = len(channels)
    N = Data.shape[-1]

    assert Data.ndim == 2 or Data.ndim == 3, "Your Data must be 3 or 2 Dimensional, (Trials (optional), Channels, Time)"

    if Data.ndim == 3:

        Data = np.squeeze(np.mean(Data, axis = 0))

    kernel = AssignOrderEstFunction(options['kernel'])

    ctr = 0
    
    orders_mat = np.zeros((number_of_channels, number_of_channels))

    orders = np.arange(min_order, max_order, leap_length)
    
    for a, channel_a in enumerate(channels):

        for b, channel_b in enumerate(channels):

            if a != b:

                BICs = []

                x = Data[channel_a, :]
                y = Data[channel_b, :]

                for order in orders:

                    pred_err = kernel(x, y, order)
                    
                    BICs.append(BIC_calc(pred_err, N, order))

                orders_mat[a, b] = int(KneeLocator(orders, BICs, curve = 'convex', direction = 'decreasing').knee)
 
    return orders_mat

