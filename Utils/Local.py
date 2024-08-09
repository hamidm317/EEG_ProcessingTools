import h5py
from pandas import read_excel, read_csv

from Utils.Constants import LocalDataConstants

def ClusteredEEGLoader(event_number):

    eeg_file_dir = LocalDataConstants.directories['eeg_file_dir']

    # events = ['All', 'Neg', 'Pos', 'Stim']

    with h5py.File(eeg_file_dir, 'r') as f:

        raw_data = f['All_data_' + str(LocalDataConstants.names['events'][event_number])][:]

    with h5py.File(eeg_file_dir, 'r') as f:

        data_lengths = f['data_lengths'][event_number, :]
        
    raw_data = raw_data.transpose(3, 1, 2, 0)

    return raw_data, data_lengths

def ExperimentDataLoader():

    BehavioralData = read_excel(LocalDataConstants.directories['beh_dir_file'])
    Performance_data = read_csv(LocalDataConstants.directories['perform_data_dir'])

    return BehavioralData, Performance_data

def AvailableSubjects():

    BehavioralData, tmp = ExperimentLoader()

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