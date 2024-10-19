import numpy as np
import pickle
import scipy.stats as sps

import Modules.GRConnPy as GRC
import Modules.GRUniPy as GRU
from Utils import Local
from Utils import Constants
from Utils import SciPlot as SP

from tqdm import tqdm
from Utils.InputVariables import CalculationVars as CV
from Utils.InputVariables import CommonVars as CoV

########################################################### Define Parameters ###########################################################

DataName = CV.NetBaseCoModCalc['DataName']
NodeNames = Constants.LocalDataConstants.names[DataName + 'ClusterNames']

group_labels = Constants.LocalDataConstants.Labels['groups']
data_labels = Constants.LocalDataConstants.Labels['data_block']

ConKers = CV.NetBaseCoModCalc['ConKers']
overlap_ratio = CV.NetBaseCoModCalc['OLR']
win_length = CV.NetBaseCoModCalc['WinLen']
NOIs = CV.NetBaseCoModCalc['NOIs']
Bands = CV.NetBaseCoModCalc['Bands']


FilterInKernel = CV.NetBaseCoModCalc['FilterInKernel']

OutSource = CV.NetBaseCoModCalc['OutSource']

st = CV.NetBaseCoModCalc['st']
ft = CV.NetBaseCoModCalc['ft']

Fs = CoV.SamplingFrequency

confile_dir = Constants.LocalDataConstants.directories['n_confile_dir']

sp = int((st + 1) * Fs)
fp = int((ft + 1) * Fs)

NB = CV.NetBaseCoModCalc['NB']
TB = CV.NetBaseCoModCalc['TB']

########################################################### Load Available Data ###########################################################

BehavioralData, _ = Local.ExperimentDataLoader()
SOI = Local.AvailableSubjects()

########################################################## Generate Connectivity Data ###########################################################

event_numbers = CV.NetBaseCoModCalc['Events']

specs = {

    'orders_matrix': CV.NetBaseCoModCalc['OrderMat'],
    'overlap_ratio': overlap_ratio,
    'window_length': win_length,
    'start time': st,
    'end time': ft,
    'DecompKern': CV.NetBaseCoModCalc['DecompKern'],
    'CorrCalcFunct': CV.NetBaseCoModCalc['CorrCalcFunct'],
    'AmpBand': CV.NetBaseCoModCalc['AmpBand'],
    'PhaseBand': CV.NetBaseCoModCalc['PhaseBand'],
}

for event in event_numbers:

    event_name = Constants.LocalDataConstants.names['events'][event]

    raw_data, data_lengths = Local.ClusteredEEGLoader(event = event_name, data_name = CV.NetBaseCoModCalc['DataName'])
    
    print("The Event is " + event_name)

    for NOI in NOIs:

        for kernel in ConKers:

            specs['Kernel'] = kernel

            for Band_i, Band in enumerate(Bands):

                if Local.BandAvailable(kernel, Band):

                    SaveFileDir = Local.HandleDir(confile_dir + '\\CommonEraData\\' + DataName + "\\" + event_name + '\\' + NOI + '\\CoMod\\' + kernel + '\\' + Band)

                    tConDataDict = {}

                    for i, sub_i in tqdm(enumerate(SOI[0])):

                        print("subject " + str(i))

                        dl = int(data_lengths[i])
                        data = sps.zscore(raw_data[i][:, :, sp : fp], axis = -1)

                        if NB == 1:

                            Samples = SP.RandomBlockSampling(dl, NumBlock = NB, NumSample_inBlock = TB)

                        else:

                            Samples = SP.DeterminedBlockSampling(dl, NumBlock = NB, NumSample_inBlock = TB)

                        divData = np.mean(data[Samples], axis = 1)

                        if Band != 'All' and not FilterInKernel:

                            divData = GRU.FrequencyBandExt(divData, Band = Band)
                            Band_ = 'All'

                        else:

                            Band_ = Band

                        sub_Data = GRC.DynamicComodulogram(divData, kernel = kernel, Band = Band_, OutSource = OutSource, inc_channels = Constants.LocalDataConstants.NetworksOfInterest[DataName][NOI], **specs)
                        
                        tConDataDict[str(SOI[1][i])] = sub_Data

                    SaveFileName, version_number = Local.HandleFileName(SaveFileDir, specs)

                    with open(SaveFileDir + "\\" + SaveFileName, 'wb') as f:
                    
                        pickle.dump(tConDataDict, f, protocol=pickle.HIGHEST_PROTOCOL)

                    print("Version " + str(version_number) + " of File Saved")

                    SaveFileDir = Local.HandleDir(confile_dir + '\\' + event_name)