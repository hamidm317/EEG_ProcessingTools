import numpy as np
import pickle
import scipy.stats as sps

import Modules.GRConnPy as GRC
from Utils import Local
from Utils import Constants
from Utils import SciPlot as SP

from tqdm import tqdm

from Utils.InputVariables import CalculationVars as CV
from Utils.InputVariables import CommonVars as CoV

########################################################### Define Parameters ###########################################################

group_labels = Constants.LocalDataConstants.Labels['groups']
data_labels = Constants.LocalDataConstants.Labels['data_block']

ConKers = CV.NetBaseConnSigTest['ConKers']
overlap_ratio = CV.NetBaseConnSigTest['OLR']
win_length = CV.NetBaseConnSigTest['WinLen']
NOIs = CV.NetBaseConnSigTest['NOIs']
Bands = CV.NetBaseConnSigTest['Bands']

OutSource = CV.NetBaseConnSigTest['OutSource']

st = CV.NetBaseConnSigTest['st']
ft = CV.NetBaseConnSigTest['ft']

Fs = CoV.SamplingFrequency

confile_dir = Constants.LocalDataConstants.directories['n_confile_dir']

sp = int((st + 0.4) * Fs)
fp = int((ft + 0.4) * Fs)

NB = ft = CV.NetBaseConnSigTest['NB']
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

event_numbers = [3] # Stim Onset

event_numbers = CV.NetBaseConnSigTest['Events']

specs = {

    'orders_matrix': CV.NetBaseConnSigTest['OrderMat'],
    'overlap_ratio': overlap_ratio,
    'window_length': win_length,
    'start time': st,
    'end time': ft,
    'DecompKern': CV.NetBaseConnSigTest['DecompKern'],
    'CorrCalcFunct': CV.NetBaseConnSigTest['CorrCalcFunct'],
    'AmpBand': CV.NetBaseConnSigTest['AmpBand'],
    'PhaseBand': CV.NetBaseConnSigTest['PhaseBand'],
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

                    for i, sub_i in tqdm(enumerate(SOI[0])):

                        print("subject " + str(i))

                        dl = int(data_lengths[i])
                        data = sps.zscore(raw_data[i][:, :, sp : fp], axis = -1)
                        Samples = SP.DeterminedBlockSampling(dl, NumBlock = NB, NumSample_inBlock = TB)
                        divData = np.mean(data[Samples], axis = 1)

                        sub_Data = GRC.ConnectionSignificance(divData, kernel = kernel, Band = Band, overlap_ratio = specs['overlap_ratio'], window_length = specs['window_length'], orders_matrix = specs['orders_matrix'], inc_channels = Constants.LocalDataConstants.NetworksOfInterest[NOI], AmpBand = 'Gamma', PhaseBand = 'Theta', PhaseAmplitudeCorrelateCalc = 'ModulationIndex')
                        
                        tConDataDict[str(SOI[1][i])] = sub_Data

                    SaveFileName, version_number = Local.HandleFileName(SaveFileDir, specs)

                    with open(SaveFileDir + "\\" + SaveFileName, 'wb') as f:
                    
                        pickle.dump(tConDataDict, f, protocol=pickle.HIGHEST_PROTOCOL)

                    print("Version " + str(version_number) + " of File Saved")

                    SaveFileDir = Local.HandleDir(confile_dir + '\\SignificanceTest_' + event_name)