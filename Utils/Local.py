import h5py
from pandas import read_excel, read_csv
import numpy as np
import os
import pandas as pd
from datetime import datetime

import pickle
from scipy.io import loadmat

from Utils.Constants import LocalDataConstants, DC_Constants
from Utils import DataLoadUtils as DLU

def ClusteredEEGLoader(event, data_name = 'July'):

    EEGLoatDataFunction = getattr(DLU, data_name + 'EEGDataLoad')

    raw_data, data_lengths = EEGLoatDataFunction(event)

    return raw_data, data_lengths

def ExperimentDataLoader():

    BehavioralData = read_excel(LocalDataConstants.directories['beh_dir_file'])
    Performance_data = read_csv(LocalDataConstants.directories['perform_data_dir'])

    return BehavioralData, Performance_data

def AvailableSubjects():

    SOI = np.load(LocalDataConstants.directories['ListOfAvailableSubjects'])

    return SOI

def HandleDir(Directory):

    if not os.path.isdir(Directory):

        os.makedirs(Directory)

    return Directory

def HandleFileName(SaveFileDir, specs):

    CurrentTime = datetime.now()

    if not os.path.isfile(SaveFileDir + "\\VersionHistory.csv"):

        if specs['Kernel'] != 'PAC':

            HistoryDict = {'idx': 0, 'VersionNumber': [0], 'Date': [str(CurrentTime.year) + '-' + str(CurrentTime.month) + '-' + str(CurrentTime.day)], 
                        'Window_Length': [specs['window_length']], 'Overlap_Ratio': [specs['overlap_ratio']], 'Start Time': [specs['start time']],
                        'End Time': [specs['end time']],
                        'OrdersMatrix': [specs['orders_matrix']]}
            
        else:
            
            HistoryDict = {'idx': 0, 'VersionNumber': [0], 'Date': [str(CurrentTime.year) + '-' + str(CurrentTime.month) + '-' + str(CurrentTime.day)], 
                        'Window_Length': [specs['window_length']], 'Overlap_Ratio': [specs['overlap_ratio']], 'Start Time': [specs['start time']],
                        'End Time': [specs['end time']],
                        'OrdersMatrix': [specs['orders_matrix']], 'AmpBand': [specs['AmpBand']], 'PhaBand': [specs['PhaseBand']]}

        DF = pd.DataFrame(HistoryDict)

        DF.to_csv(SaveFileDir + "\\VersionHistory.csv", index = False)

        version_number = 0

    else:

        HistoryDict = read_csv(SaveFileDir + "\\VersionHistory.csv", index_col = 0)

        version_number = HistoryDict['VersionNumber'][len(HistoryDict['VersionNumber']) - 1] + 1

        if specs['Kernel'] != 'PAC':

            new_row = [version_number, str(CurrentTime.year) + '-' + str(CurrentTime.month) + '-' + str(CurrentTime.day), specs['window_length'],
                    specs['overlap_ratio'], specs['start time'], specs['end time'], specs['orders_matrix']]
            
        else:

            new_row = [version_number, str(CurrentTime.year) + '-' + str(CurrentTime.month) + '-' + str(CurrentTime.day), specs['window_length'],
                    specs['overlap_ratio'], specs['start time'], specs['end time'], specs['orders_matrix'], specs['AmpBand'], specs['PhaseBand']]
        
        HistoryDict.loc[len(HistoryDict['VersionNumber'])] = new_row
        HistoryDict.sort_index()

        HistoryDict.to_csv(SaveFileDir + "\\VersionHistory.csv")

    SaveFileName = "Data_Version" + str(version_number)

    return SaveFileName, version_number

def BandAvailable(Kernel, Band):

    if Band in DC_Constants.Properties[Kernel]['AvailableBands']:

        return True
    
    else:

        return False
    
def HandleDataLoad(Dir, version_number = None):

    assert os.path.isdir(Dir), "This Data is not Available"

    VersionHistoryDF = read_csv(Dir + "\\VersionHistory.csv", index_col = 0)

    if version_number is None:

        version_number = np.array(VersionHistoryDF['VersionNumber'])[-1]

    LoadFileDir = Dir + "\\Data_Version" + str(version_number)

    with open(LoadFileDir, 'rb') as f:

        ConDataDict = pickle.load(f)

    DataSpecs = {key_SD: VersionHistoryDF[key_SD][version_number] for key_SD in VersionHistoryDF.keys()}

    return ConDataDict, DataSpecs