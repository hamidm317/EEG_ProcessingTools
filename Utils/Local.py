import h5py
from pandas import read_excel, read_csv
import numpy as np
import os
import pandas as pd
from datetime import datetime

import pickle

from Utils.Constants import LocalDataConstants, DC_Constants

def ClusteredEEGLoader(event):

    if event != 'PosNeg':

        eeg_file_dir = LocalDataConstants.directories['eeg_file_dir']

        assert type(event) == int or type(event) == str, "The 'event' must be the event name as string or event number as integer"

        if type(event) == str:

            tmp = [LocalDataConstants.names['events'][i] == event for i in range(len(LocalDataConstants.names['events']))]

            assert np.any(tmp), "The Event is not available"

            event_number = np.where(tmp)[0][0]

        else:

            assert event >= 0 and event < len(LocalDataConstants.names['events']), "The Event is not available"
            
            event_number = event

        with h5py.File(eeg_file_dir, 'r') as f:

            raw_data = f['All_data_' + str(LocalDataConstants.names['events'][event_number])][:]

        with h5py.File(eeg_file_dir, 'r') as f:

            data_lengths = f['data_lengths'][event_number, :]
            
        raw_data = raw_data.transpose(3, 1, 2, 0)

    else: # Pure Shit, CLEAN THIS SHIT

        events = ['Pos', 'Neg']

        raw_data = []
        data_lengths = []

        for event in events:

            tmp_raw_data, tmp_data_length = ClusteredEEGLoader(event)

            raw_data.append(tmp_raw_data)
            data_lengths.append(tmp_data_length)

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

        HistoryDict = {'idx': 0, 'VersionNumber': [0], 'Date': [str(CurrentTime.year) + '-' + str(CurrentTime.month) + '-' + str(CurrentTime.day)], 
                       'Window_Length': [specs['window_length']], 'Overlap_Ratio': [specs['overlap_ratio']], 'Start Time': [specs['start time']],
                       'End Time': [specs['end time']],
                       'OrdersMatrix': [specs['orders_matrix']]}
        DF = pd.DataFrame(HistoryDict)

        DF.to_csv(SaveFileDir + "\\VersionHistory.csv", index = False)

        version_number = 0

    else:

        HistoryDict = read_csv(SaveFileDir + "\\VersionHistory.csv", index_col = 0)

        version_number = HistoryDict['VersionNumber'][len(HistoryDict['VersionNumber']) - 1] + 1

        new_row = [version_number, str(CurrentTime.year) + '-' + str(CurrentTime.month) + '-' + str(CurrentTime.day), specs['window_length'],
                   specs['overlap_ratio'], specs['start time'], specs['end time'], specs['orders_matrix']]
        
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