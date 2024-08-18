import h5py
from pandas import read_excel, read_csv
import numpy as np

from Utils.Constants import LocalDataConstants

def ClusteredEEGLoader(event):

    if event != 'PosNeg':

        eeg_file_dir = LocalDataConstants.directories['eeg_file_dir']

        assert type(event) == int or type(event) == str, "The 'event' must be the event name as string or event number as integer"

        if type(event) == str:

            tmp = [LocalDataConstants.names['events'][i] == event for i in range(len(LocalDataConstants.names['events']))]

            assert np.any(tmp), "The Event is not available"

            event_number = np.where(tmp)[0][0]

        else:

            assert event > 0 and event < len(LocalDataConstants.names['events']), "The Event is not available"
            
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

    BehavioralData, tmp = ExperimentDataLoader()

    import os

    subjects = []

    for path in os.listdir(LocalDataConstants.directories['eeg_prep_datasets_dir']):

        subjects.append(path)

    subjects_of_interest = []  
    available_IDs = []                      
                
    for i in range(len(subjects)):

        idx = np.where((BehavioralData['id'][:]) == int(subjects[i]))[0][0]
        subjects_of_interest.append(idx)

        available_IDs.append(BehavioralData['id'][idx])

    return np.array(subjects_of_interest), available_IDs