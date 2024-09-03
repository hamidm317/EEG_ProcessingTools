import numpy as np
import pickle
import scipy.stats as sps

import Modules.GRConnPy as GRC
from Utils import Local
from Utils import Constants
from Utils import SciPlot as SP

########################################################### Define Parameters ###########################################################

group_labels = Constants.LocalDataConstants.Labels['groups']
data_labels = Constants.LocalDataConstants.Labels['data_block']

# ConKers = [kernel for kernel in Constants.DC_Constants.Properties.keys()]
ConKers = ['PLI', 'TE', 'dPLI', 'wPLI']

overlap_ratio = Constants.LocalDataConstants.DefaulValues['overlap_ratio']
win_length = Constants.LocalDataConstants.DefaulValues['window_length']

confile_dir = Constants.LocalDataConstants.directories['n_confile_dir']

# NOIs = [Network for Network in Constants.LocalDataConstants.NetworksOfInterest.keys()][1:] # Networks Of Interest!
NOIs = ['ZeroAxis', 'Frontal', 'OcciTemporal']
Bands = Constants.LocalDataConstants.names['freq_bands']

Fs = 500

st = -0.2
ft = 0.6
sp = int((st + 0.4) * Fs)
fp = int((ft + 0.4) * Fs)

NB = 2
TB = Constants.LocalDataConstants.DefaulValues['trial_in_block']

########################################################### Load Available Data ###########################################################

BehavioralData, _ = Local.ExperimentDataLoader()
SOI = Local.AvailableSubjects()

################################################## Divide Subject into DEP and CTRL Groups ###########################################################

Sub_G = [[], []] # first element is CTRL Group Members and the Second one the DEP Group

for i, sub_i in enumerate(SOI[0]):

    if BehavioralData['BDI'][sub_i] < 10:

        Sub_G[0].append([i, sub_i])

    else:

        Sub_G[1].append([i, sub_i])

########################################################## Generate Connectivity Data ###########################################################

event_numbers = [0] # Stim Onset

specs = {

    'orders_matrix': 12,
    'overlap_ratio': overlap_ratio,
    'window_length': win_length,
    'start time': st,
    'end time': ft
}

for event in event_numbers:

    raw_data, data_lengths = Local.ClusteredEEGLoader(event = event)
    
    event_name = Constants.LocalDataConstants.names['events'][event]
    print("The Event is " + event_name)

    SaveFileDir = Local.HandleDir(confile_dir + '\\' + event_name)

    for NOI in NOIs:

        SaveFileDir = Local.HandleDir(confile_dir + '\\' + event_name + '\\' + NOI)

        for kernel in ConKers:

            SaveFileDir = Local.HandleDir(confile_dir + '\\' + event_name + '\\' + NOI + '\\' + kernel)

            for Band_i, Band in enumerate(Bands):

                if Local.BandAvailable(kernel, Band):

                    SaveFileDir = Local.HandleDir(confile_dir + '\\' + event_name + '\\' + NOI + '\\' + kernel + '\\' + Band)

                    tConDataDict = {}

                    for i, sub_i in enumerate(SOI[0]):

                        print("subject " + str(i))

                        dl = int(data_lengths[i])
                        data = sps.zscore(raw_data[i][:, :, sp : fp], axis = -1)
                        Samples = SP.DeterminedBlockSampling(dl, NumBlock = NB, NumSample_inBlock = TB)
                        divData = np.mean(data[Samples], axis = 1)

                        sub_Data = GRC.DynamicConnectivityMeasure(divData, kernel = kernel, Band = Band, overlap_ratio = specs['overlap_ratio'], window_length = specs['window_length'], orders_matrix = specs['orders_matrix'], inc_channels = Constants.LocalDataConstants.NetworksOfInterest[NOI])
                        
                        tConDataDict[str(SOI[1][i])] = sub_Data

                    SaveFileName, version_number = Local.HandleFileName(SaveFileDir, specs)

                    with open(SaveFileDir + "\\" + SaveFileName, 'wb') as f:
                    
                        pickle.dump(tConDataDict, f, protocol=pickle.HIGHEST_PROTOCOL)

                    print("Version " + str(version_number) + " of File Saved")

                    SaveFileDir = Local.HandleDir(confile_dir + '\\' + event_name)