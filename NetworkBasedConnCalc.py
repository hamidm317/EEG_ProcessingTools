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

DataName = CV.NetBaseConnCalc['DataName']
NodeNames = Constants.LocalDataConstants.names[DataName + 'ClusterNames']

group_labels = Constants.LocalDataConstants.Labels['groups']
data_labels = Constants.LocalDataConstants.Labels['data_block']

ConKers = CV.NetBaseConnCalc['ConKers']
overlap_ratio = CV.NetBaseConnCalc['OLR']
win_length = CV.NetBaseConnCalc['WinLen']
NOIs = CV.NetBaseConnCalc['NOIs']
Bands = CV.NetBaseConnCalc['Bands']

CommonEraData = CV.NetBaseConnCalc['CommonEraData']

FilterInKernel = CV.NetBaseConnCalc['FilterInKernel']

OutSource = CV.NetBaseConnCalc['OutSource']

st = CV.NetBaseConnCalc['st']
ft = CV.NetBaseConnCalc['ft']

Fs = CoV.SamplingFrequency

data_name = CV.NetBaseConnCalc['DataName']

confile_dir = Constants.LocalDataConstants.directories['n_confile_dir']

if CommonEraData:

    confile_dir = confile_dir + "\\CommonEraData"

NB = CV.NetBaseConnCalc['NB']
TB = CV.NetBaseConnCalc['TB']

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

event_numbers = CV.NetBaseConnCalc['Events']

specs = {

    'orders_matrix': CV.NetBaseConnCalc['OrderMat'],
    'overlap_ratio': overlap_ratio,
    'window_length': win_length,
    'start time': st,
    'end time': ft,
    'DecompKern': CV.NetBaseConnCalc['DecompKern'],
    'CorrCalcFunct': CV.NetBaseConnCalc['CorrCalcFunct'],
    'AmpBand': CV.NetBaseConnCalc['AmpBand'],
    'PhaseBand': CV.NetBaseConnCalc['PhaseBand'],

    'SingleTrial': CV.NetBaseConnCalc['SingleTrial'],
}

for event in event_numbers:

    event_name = Constants.LocalDataConstants.names['events'][event]

    if event_name == 'Actions':

        sp = int((st + 1) * Fs)
        fp = int((ft + 1) * Fs)

    else:

        sp = int((st + 0.4) * Fs)
        fp = int((ft + 0.4) * Fs)

    raw_data, data_lengths = Local.ClusteredEEGLoader(event = event_name, data_name = data_name)
    
    print("The Event is " + event_name)

    for NOI in NOIs:

        for kernel in ConKers:

            specs['Kernel'] = kernel

            for Band_i, Band in enumerate(Bands):

                if Local.BandAvailable(kernel, Band):

                    # SaveFileDir = Local.HandleDir(confile_dir + '\\' + data_name + "\\" + event_name + '\\' + NOI + '\\' + kernel + '\\' + Band)
                    SaveFileDir = Local.SaveDirGen({'MainDir': confile_dir, 'CurrEventName': event_name, 'Network': NOI, 'KernelName': kernel, 'BandName': Band, 'DataName': data_name, 'Specs': specs})

                    tConDataDict = {}

                    for i, sub_i in tqdm(enumerate(SOI[0])):

                        print("subject " + str(i))

                        dl = int(data_lengths[i])
                        data = sps.zscore(raw_data[i][:, :, sp : fp], axis = -1)

                        if NB == 1:

                            Samples = SP.RandomBlockSampling(dl, NumBlock = NB, NumSample_inBlock = TB)

                        else:

                            Samples = SP.DeterminedBlockSampling(dl, NumBlock = NB, NumSample_inBlock = TB)

                        divData = data[Samples]

                        if CV.NetBaseConnCalc['SingleTrial']:

                            BDC_Data = []
                            sub_Data = []

                            for dataBlock in divData:

                                for SingleTrialData in dataBlock:

                                    tmpDC_Data = GRC.DynamicConnectivityMeasure(SingleTrialData, kernel = kernel, Band = Band, OutSource = OutSource, inc_channels = Constants.LocalDataConstants.NetworksOfInterest[DataName][NOI], **specs)

                                    BDC_Data.append(tmpDC_Data)

                                sub_Data.append(np.mean(np.array(BDC_Data), axis = 0))

                        else:

                            divData = np.mean(divData, axis = 1)

                            if Band != 'All' and not FilterInKernel:

                                divData = np.squeeze(np.array([GRU.FrequencyBandExt(divData[i], Band = Band) for i in range(divData.shape[0])]))
                                Band_ = 'All'

                            else:

                                Band_ = Band

                            sub_Data = GRC.DynamicConnectivityMeasure(divData, kernel = kernel, Band = Band_, OutSource = OutSource, inc_channels = Constants.LocalDataConstants.NetworksOfInterest[DataName][NOI], **specs)
                        
                        tConDataDict[str(SOI[1][i])] = np.array(sub_Data)

                    SaveFileName, version_number = Local.HandleFileName(SaveFileDir, specs)

                    with open(SaveFileDir + "\\" + SaveFileName, 'wb') as f:
                    
                        pickle.dump(tConDataDict, f, protocol=pickle.HIGHEST_PROTOCOL)

                    print("Version " + str(version_number) + " of File Saved")

                    SaveFileDir = Local.HandleDir(confile_dir + '\\' + event_name)