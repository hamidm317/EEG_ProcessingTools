import numpy as np
from kneed import KneeLocator
from sklearn.linear_model import LinearRegression as LR
import matplotlib.pyplot as plt
from pandas import read_excel, read_csv

import scipy.signal as scs

import pywt

import h5py

####################################################### General Functions #############################################################

def channels_specs(file_loc):
    
    file = open(file_loc, 'r')

    raw_file = file.read()
    channel_name = raw_file.replace('\t', ' ').split("\n")

    file.close()
    
    return channel_name


def AIC_calc(y, y_pred, k):
    
    # AIC = 2k + n * ln(mean sum of residuals)
    
    n = len(y)
    
    if len(y) != len(y_pred):
        
        print("Predicted values and real data doesn't have same length")
        
        return ''
    
    MSR = np.sum((y - y_pred) ** 2) / n
    
    return 2 * k + n * np.log(MSR)

def BIC_calc(y, y_pred, k):
    
    # BIC = k * ln(n) + n * ln(mean sum of residuals)
    
    n = len(y)
    
    if len(y) != len(y_pred):
        
        print("Predicted values and real data doesn't have same length")
        
        return ''
    
    MSR = np.sum((y - y_pred) ** 2) / n
    
    return k * np.log(n) + n * np.log(MSR)


def movmean(A, k):

    win_length = k
    MAFA = []
    
    if len(A) < k:

        print("I don't know how to calculate this, The window must be less than length. So the length forced to be len(A)")

        win_length = len(A)

    new_elements_len = len(A) - win_length + 1

    for element in range(new_elements_len):

        MAFA.append(np.mean(A[element : (element + 1) + k ]))

    return np.array(MAFA)

def Accuracy_Performance(X):

    Acc = []

    for i in range(len(X)):

        Acc.append(1 - np.sum(X[:i]) / (i + 1))

    return Acc

def roll_mat_gen(x, k):

    assert x.ndim == 1, "x must be a vector"

    X_rm = []

    N = len(x)

    for i in range(N - k):

        X_rm.append(x[i : i + k])

    return np.array(X_rm)

def RandomBlockSampling(Length: int, NumBlock: int, NumSample_inBlock: int):

    Block_Len = int(Length / NumBlock)

    assert NumSample_inBlock <= Block_Len, 'Number of Samples in Blocks must be less than length of Block'

    SampleMat = []

    for Block in range(NumBlock - 1):

        SampleMat.append(np.random.permutation(Block_Len)[:NumSample_inBlock] + Block * Block_Len)

    SampleMat.append(Length - 1 - np.random.permutation(Block_Len)[:NumSample_inBlock])

    return np.array(SampleMat)

def DeterminedBlockSampling(Length: int, NumBlock: int, NumSample_inBlock: int):

    Block_Len = int(Length / NumBlock)

    assert NumSample_inBlock <= Block_Len, 'Number of Samples in Blocks must be less than length of Block'

    SampleMat = []

    for Block in range(NumBlock - 1):

        SampleMat.append(np.int32(np.arange(NumSample_inBlock)) + Block * Block_Len)

    SampleMat.append(Length - 1 - np.int32(np.arange(NumSample_inBlock)))

    return np.array(SampleMat)

def PhaseDecomp(Data: np.ndarray, NumBlock: int, NumSample_inBlock: int, SamplingMethod = 'DET'):
    
    assert Data.ndim == 1, 'Trend must be a vector'

    Length = len(Data)

    if SamplingMethod == 'STO':

        SamplesInBlocks = RandomBlockSampling(Length, NumBlock, NumSample_inBlock)

    else:

        SamplesInBlocks = DeterminedBlockSampling(Length, NumBlock, NumSample_inBlock)
    
    return np.mean(Data[SamplesInBlocks], axis = 1)

def generate_confidenceBounds(Data, Confidence_Level = 0.98):

    assert Data.ndim < 3 and Data.ndim > 0, "Data must be vector or 2 way matrix"

    if Data.ndim == 2:

        y_Upper = []
        y_Lower = []

        [num_samples, num_timePoint] = Data.shape

        for t_p in range(num_timePoint):

            U_tmp, L_tmp = ConfidenceBound(Data[:, t_p], Confidence_Level)

            y_Upper.append(L_tmp)
            y_Lower.append(U_tmp)

        return np.array(y_Upper), np.array(y_Lower)
    
    else:

        return 0
    
def ConfidenceBound(Data, Confidence_Level):

    UB_CI = np.mean(Data) + Confidence_Level * np.std(Data)
    LB_CI = np.mean(Data) - Confidence_Level * np.std(Data)

    return UB_CI, LB_CI

####################################################### Granger Causality #############################################################

def OrderEstimate_byChannels(Data, channels, max_order, min_order, leap_length):

    # This channel estimate the order of AR process between channels,
    # Data -> m-by-n matrix, m is number of channels and n is length of data,
    # channels -> channels of Data matrix which orders must be calculated (list with maximum length m),
    # max_order -> maximum valid order to be included in estimation (less than n)
    # min_order -> minimum valid order to be included in estimation (more than zero)
    # leap_length -> orders likelihood will be computed with this leap_length
    
    number_of_channels = len(channels)

    ctr = 0
    
    orders_mat = np.zeros((number_of_channels, number_of_channels))

    orders = np.arange(min_order, max_order, leap_length)
    
    for a, channel_a in enumerate(channels):

        for b, channel_b in enumerate(channels):

            if a != b:

                BICs = []

                x_t = Data[channel_a, :]
                y_t = Data[channel_b, :]

                for order in orders:

                    a_est_mul, b_est_mul = mulvar_AR_est(x_t, y_t, order, order)

                    x_t_rec_mat = x_t_recun_ab(a_est_mul, b_est_mul, x_t, y_t)
                    BICs.append(BIC_calc(x_t, x_t_rec_mat, order))
                    
                # plt.plot(BICs)
                # plt.show()
                
                ctr = ctr + 1
                
                # orders_mat[a, b] = orders[int(np.argmin(BICs))]
                orders_mat[a, b] = int(KneeLocator(orders, BICs, curve = 'convex', direction = 'decreasing').knee)
                print(int(ctr / (number_of_channels * (number_of_channels - 1)) * 100), "%", "Estimated Order is", orders_mat[a, b])
 
    return orders_mat

def a_estimation_err(a_est, x_t):
    
    order = len(a_est)
    length = len(x_t)
    
    x_t_rec = np.zeros(length)
    x_t_rec[:order] = x_t[:order]

    for i in range(order, length):

        for j in range(order):

            x_t_rec[i] = x_t_rec[i] + a_est[j] * x_t[i - j - 1]

    return np.sum((x_t - x_t_rec) ** 2)

def x_t_recun(a_est, x_t):
    
    order = len(a_est)
    length = len(x_t)
    
    x_t_rec = np.zeros(length)
    x_t_rec[:order] = x_t[:order]

    for i in range(order, length):

        for j in range(order):

            x_t_rec[i] = x_t_rec[i] + a_est[j] * x_t[i - j - 1]

    return x_t_rec

def ab_estimation_err(a_est, b_est, x_t, y_t):
    
    a_order = len(a_est)
    b_order = len(b_est)
    
    length = len(x_t)
    
    x_t_rec = np.zeros(length)
    x_t_rec[:a_order] = x_t[:a_order]

    for i in range(a_order, length):

        for j in range(a_order):

            x_t_rec[i] = x_t_rec[i] + a_est[j] * x_t[i - j - 1]
            
        for j in range(b_order):

            x_t_rec[i] = x_t_rec[i] + b_est[j] * y_t[i - j - 1]

    return np.sum((x_t - x_t_rec) ** 2)

def x_t_recun_ab(a_est, b_est, x_t, y_t):
    
    a_order = len(a_est)
    b_order = len(b_est)
    
    length = len(x_t)
    
    x_t_rec = np.zeros(length)
    x_t_rec[:a_order] = x_t[:a_order]

    for i in range(a_order, length):

        for j in range(a_order):

            x_t_rec[i] = x_t_rec[i] + a_est[j] * x_t[i - j - 1]
            
        for j in range(b_order):

            x_t_rec[i] = x_t_rec[i] + b_est[j] * y_t[i - j - 1]

    return x_t_rec

def GC_calc(x_t, y_t, order):

    # it is much better to give access to orders to users!
    GC_val = []
    univar_error = []
    mulvar_error = []

    a_est_uni = univar_AR_est(x_t, order)

    a_order = order
    b_order = a_order

    a_est_mul, b_est_mul = mulvar_AR_est(x_t, y_t, a_order, b_order)

    univar_error.append(a_estimation_err(a_est_uni, x_t))
    mulvar_error.append(ab_estimation_err(a_est_mul, b_est_mul, x_t, y_t))
        
    GC_val.append(np.log(univar_error[-1] / mulvar_error[-1]))
        
    # print("Order is", order, "and Granger Causality is", np.log(a_estimation_err(a_est_uni, x_t) / ab_estimation_err(a_est_mul, b_est_mul, x_t, y_t)), "Univar Error is", a_estimation_err(a_est_uni, x_t), "and mulvar error is", ab_estimation_err(a_est_mul, b_est_mul, x_t, y_t))
        
    return GC_val, univar_error, mulvar_error

def GrangerCausalityEstimator(Data, channels, window_length, overlap_ratio, orders_mat, core_method = 'pinv'):

    # # This Function calculates the Granger Causality between different channels of Data
    # Data -> m-by-n data matrix, m is number of channels and n is length of data
    # channels -> list of channels to calculate GC
    # window_length -> Length of windows to compute GC over them
    # overlap_ratio -> Precent of overlap between consequent windows
    # orders_mat -> order of AR process estimation between each two channel (must be a m-by-m channel)
    # it is considered to be output of function 'OrderEstimate_byChannels' above
    
    N = Data.shape[-1]
    number_of_channels = len(channels)
    number_of_windows = int((N - window_length) / ((1 - overlap_ratio) * window_length)) + 1

    GC_values = np.zeros((number_of_windows, number_of_channels, number_of_channels))

    for win_step in range(number_of_windows):

        print("In Progress", win_step / number_of_windows * 100, "% ...")

        for i, channel_a in enumerate(channels):

            for j, channel_b in enumerate(channels):

                if i != j:

                    win_stp = int((win_step) * (1 - overlap_ratio) * window_length)
                    win_enp = win_stp + window_length

                    x_t = Data[channel_a, win_stp : win_enp]
                    y_t = Data[channel_b, win_stp : win_enp]

                    est_order = int(orders_mat[i, j])

                    if core_method == 'LRB':

                        win_GC_val = LRB_GC_calc(x_t, y_t, est_order)

                    else:

                        tmp, tmp_err1, tmp_err2 = GC_calc(x_t, y_t, est_order)

                        win_GC_val = tmp[0]

                    GC_values[win_step, i, j] = win_GC_val
                    
    return GC_values


def LRB_univar_e(x, k):

    assert x.ndim == 1, "x must be a vector"

    X_rm = roll_mat_gen(x, k)

    X_des = x[k:]

    LR_M = LR().fit(X_rm, X_des)

    return 1 - LR_M.score(X_rm, X_des)

def LRB_mulvar_e(x, y, k):

    assert x.ndim == 1 and y.ndim == 1, "x and y must be a vector"

    X_rm = roll_mat_gen(x, k)
    Y_rm = roll_mat_gen(y, k)

    XY = np.concatenate([X_rm.T, Y_rm.T]).T

    X_des = x[k:]

    LR_M = LR().fit(XY, X_des)

    return 1 - LR_M.score(XY, X_des)

def LRB_GC_calc(x_t, y_t, est_order):

    uve = LRB_univar_e(x_t, est_order)
    mve = LRB_mulvar_e(x_t, y_t, est_order)

    return np.log(uve / mve)

################################################### Auto-Regressive Process Tools #######################################################

def univar_AR_est(x_t, order):
    
    length = len(x_t)

    X_mat = np.zeros((length - order, order))
    X_vec = np.zeros((length - order))

    for i in range(length - order):

        X_mat[i, :] = x_t[i : i + order]
        X_vec[i] = x_t[i + order]

    return np.flip(np.linalg.pinv(X_mat) @ X_vec)

def mulvar_AR_est(x_t, y_t, a_order, b_order):
    
    length = len(x_t)

    X_mat = np.zeros((length - a_order, a_order + b_order))
    X_vec = np.zeros((length - a_order))

    for i in range(length - a_order):

        X_mat[i, : a_order] = x_t[i : i + a_order]
        X_mat[i, a_order : a_order + b_order] = y_t[i : i + b_order]
        
        X_vec[i] = x_t[i + a_order]

    coef_est = np.linalg.pinv(X_mat) @ X_vec
    
    a_est = np.flip(coef_est[:a_order])
    b_est = np.flip(coef_est[a_order : a_order + b_order])
    
    return a_est, b_est

def AR_Generator(order, a_order, b_order, length):

    # This function generates a random Auto-Regressive signal
    
    invalid_output = 1

    while invalid_output:

        x_t4 = np.zeros(length)
        x_t = np.zeros(length)

        x_t4[:a_order] = np.random.normal(0, 5, size = (a_order))

        x_t[:order] = np.random.normal(0, 5, size = (order))

        a = np.random.normal(0, 0.5, size = (order))

        a_4 = np.random.normal(0, 0.5, size = (a_order))
        b = np.random.normal(0, 0.5, size = (b_order))

        for i in range(order, length):

            for n, a_i in enumerate(a):

                x_t[i] = x_t[i] + a_i * x_t[i - n - 1]


        for i in range(a_order, length):

            for n, a_4i in enumerate(a_4):

                x_t4[i] = x_t4[i] + a_4i * x_t4[i - n - 1]

            for n, b_i in enumerate(b):

                x_t4[i] = x_t4[i] + b_i * x_t[i - n - 1]

        if np.abs(x_t4[-1]) + np.abs(x_t[-1]) < 1000000:

            invalid_output = 0
            
    return x_t4, x_t

def AR_Process_Gen(init_vals, coeffs, N):
    
    # This Function Generate an Auto-Regressive process based on initial values and coefficients determined in the inputs
    # length of the signal is N, Consider this point that stability of signal depends on these inputs.
    
    if len(init_vals) != len(coeffs):
        
        print("INVALID input, order not detemined!")
        return 0
    
    order = len(init_vals)
    
    AR_signal = np.zeros(N)
    AR_signal[:order] = init_vals
    
    for i in range(order, N):
        
        for n in range(order):
            
            AR_signal[i] = AR_signal[i] + coeffs[n] * AR_signal[i - n - 1]
            
    return AR_signal

def AR_Cross_Process_Gen(init_vals, coeffs, y, orders_vec):
    
    # This Function Generate an Auto-Regressive process based on initial values and coefficients determined in the inputs
    # length of the signal is N, Consider this point that stability of signal depends on these inputs.
    
    # This function calculates AR_Cross_Process using other signals and also init_values of it self
    
    number_of_crosses, N = y.shape
    
    if len(init_vals) != len(coeffs):
        
        print("INVALID input, order not detemined!")
        return 0
    
    order = len(init_vals)
    
    AR_signal = np.zeros(N)
    AR_signal[:order] = init_vals
    
    for i in range(order, N):
        
        for n in range(order):
            
            AR_signal[i] = AR_signal[i] + coeffs[n] * AR_signal[i - n - 1]
            
        for n_cross in range(number_of_crosses):
            
            for n in range(len(orders_vec[n_cross, :])):
                
                AR_signal[i] = AR_signal[i] + orders_vec[n_cross, n] * y[n_cross, i - n - 1]
            
    return AR_signal

############################################################ U-L ######################################################################

def a_estimation_RW(x_t, order, var, max_iter):
    
    # order = 4
    a_est = np.random.normal(0, 2, size = (order))
    # var = 1
    do_it = 1
    # max_iter = 200
    iteration_i = 0
    iteration_j = 0

    while do_it:

        est_err = a_estimation_err(a_est, x_t)
        iterate = 1

        while iterate:

            a_est_ = a_est + np.random.normal(0, var / np.log(iteration_i + 2), size = (order))

            if a_estimation_err(a_est_, x_t) < est_err:

                a_est = a_est_
                iterate = 0

            elif iteration_j > max_iter * 50:

                iterate = 0
                do_it = 0

            iteration_j = iteration_j + 1

        if iteration_i > max_iter:

            do_it = 0

        iteration_i = iteration_i + 1

        print(iteration_i, a_est, est_err)
    
    return a_est

def ab_estimation_RW(x_t, y_t, a_order, b_order, var, max_iter):
    
    a_est = np.random.normal(0, 2, size = (a_order))
    b_est = np.random.normal(0, 2, size = (b_order))
    
    do_it = 1
    
    iteration_i = 0
    iteration_j = 0

    while do_it:

        est_err = ab_estimation_err(a_est, b_est, x_t, y_t)
        iterate = 1

        while iterate:

            a_est_ = a_est + np.random.normal(0, var / np.log(iteration_i + 2), size = (a_order))
            b_est_ = b_est + np.random.normal(0, var / np.log(iteration_i + 2), size = (b_order))

            if ab_estimation_err(a_est_, b_est_, x_t, y_t) < est_err:

                a_est = a_est_
                b_est = b_est_
                iterate = 0

            elif iteration_j > max_iter * 50:

                iterate = 0
                do_it = 0

            iteration_j = iteration_j + 1

        if iteration_i > max_iter:

            do_it = 0

        iteration_i = iteration_i + 1

        print(iteration_i, a_est, b_est, est_err)
    
    return a_est, b_est

########################################################## PLI Calculations ###########################################################

def PLI_calc(x, y):
    
    x_a = scs.hilbert(x)
    y_a = scs.hilbert(y)
    
    phase_signs = np.sign(np.angle(x_a) - np.angle(y_a))
    
    return np.abs(np.mean(phase_signs))

def PLI_calc_C(x, y):
    
    phase_signs = np.sign(np.angle(x) - np.angle(y))
    
    return np.abs(np.mean(phase_signs))

def GraphGenerator_PLI(Data: np.ndarray, overlap_ratio: float, win_length: int):

    assert Data.ndim == 2, "Data should include at least two channels in only one block of time (must be 2-way Matrix/Tensor)"

    [num_channels, length] = Data.shape

    win_num = int((length - win_length) / ((1 - overlap_ratio) * win_length)) + 1

    PLI_values_channel_time = np.zeros((num_channels, num_channels, win_num))

    for channel_i in range(num_channels):

        for channel_j in range(channel_i):

            PLI_values_time = []

            for win in range(win_num):

                win_stp = int((win) * (1 - overlap_ratio) * win_length)
                win_enp = win_stp + win_length

                PLI_values_time.append(PLI_calc(Data[channel_i, win_stp : win_enp], Data[channel_j, win_stp : win_enp]))

            PLI_values_channel_time[channel_i, channel_j, :] = np.array(PLI_values_time)
            PLI_values_channel_time[channel_j, channel_i, :] = np.array(PLI_values_time)

    return PLI_values_channel_time

################################################## WaveLet Decomposition ##########################################################


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


        band_range.append(np.array([np.where(freqs > 0.5)[-1][-1], np.where(freqs < 4)[0][0]]))
        band_range.append(np.array([np.where(freqs > 4)[-1][-1], np.where(freqs < 8)[0][0]]))
        band_range.append(np.array([np.where(freqs > 8)[-1][-1], np.where(freqs < 12)[0][0]]))
        band_range.append(np.array([np.where(freqs > 13)[-1][-1], np.where(freqs < 30)[0][0]]))
        band_range.append(np.array([np.where(freqs > 30)[-1][-1], np.where(freqs < 90)[0][0]]))
  

    return np.array(CWTMat_Conn), band_range, freqs

def plot_wavelet(time, data_l, wavelet, title, ax):

    widths = np.geomspace(8, 512, num = 40)
    cwtmatrs = []

    if data_l.ndim == 2:

        [Trials, Length] = data_l.shape

        for i in range(Trials):

            cwtmatr, freqs = pywt.cwt(
                data_l[i], widths, wavelet, sampling_period=np.diff(time).mean()
            )

        cwtmatrs.append(np.abs(cwtmatr[:-1, :-1]))

    else:

        cwtmatr, freqs = pywt.cwt(
            data_l, widths, wavelet, sampling_period=np.diff(time).mean()
        )

        cwtmatrs = np.abs(cwtmatr[:-1, :-1])

    cwtmatrs = np.array(cwtmatrs)
    # print(freqs)

    pcm = ax.pcolormesh(time, freqs, np.mean(cwtmatrs, axis = 0)) # , vmin = 0, vmax = vmax)
    
    ax.set_yscale("log")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    plt.colorbar(pcm, ax=ax)
    
    return ax

def FrequencyCalibration(widths: np.ndarray, time: np.ndarray, wavelet = 'morl'):

    data = np.random.normal(loc = 0, scale = 1, size = (len(time)))

    cwtmatr, freqs = pywt.cwt(data, widths, wavelet, sampling_period = np.diff(time).mean())

    return freqs

################################################## Exclusive Loading Functions ##########################################################

def ClusteredEEG_Initial_EvBa(event_number, eeg_file_dir = r'E:\HWs\Msc\Research\Research\Depression Dataset\New Datasets\Clustered_SingleTrialData_All_Neg_Pos_Stim.mat'):

    channel_names = ['PF', 'LF', 'RF', 'MFC', 'LT', 'RT', 'LFC', 'RFC', 'MPC', 'LPC', 'RPC', 'MP', 'LPO', 'RPO']
    freq_bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    events = ['All', 'Neg', 'Pos', 'Stim']

    with h5py.File(eeg_file_dir, 'r') as f:

        raw_data = f['All_data_' + str(events[event_number])][:]

    with h5py.File(eeg_file_dir, 'r') as f:

        data_lengths = f['data_lengths'][event_number, :]
        
    raw_data = raw_data.transpose(3, 1, 2, 0)

    return raw_data, data_lengths, events, freq_bands, channel_names

def NonEP_Data_Initial(beh_dir_file = r'E:\HWs\Msc\Research\Research\Depression Dataset\depression_rl_eeg\Depression PS Task\Scripts from Manuscript\Data_4_Import.xlsx',
                       perform_data_dir = r'E:\HWs\Msc\Research\Research\Depression Dataset\New Datasets\Subjects_Behavioral_datas.csv'):

    BehavioralData = read_excel(beh_dir_file)
    Performance_data = read_csv(perform_data_dir)

    return BehavioralData, Performance_data

def FindAvailableSubjects(BehavioralData, dir_path = r'E:\\HWs\Msc\\Research\\Research\\Depression Dataset\\Testing Preprocess'):

    import os

    subjects = []

    for path in os.listdir(dir_path):

        subjects.append(path)

    subjects_of_interest = []  
    available_IDs = []                      
                
    for i in range(len(subjects)):

        idx = np.where((BehavioralData['id'][:]) == int(subjects[i]))[0][0]
        subjects_of_interest.append(idx)

        available_IDs.append(BehavioralData['id'][idx])

    return np.array(subjects_of_interest), available_IDs