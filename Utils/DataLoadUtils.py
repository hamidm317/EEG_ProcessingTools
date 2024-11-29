import h5py
import numpy as np
from scipy.io import loadmat

from Utils.Constants import LocalDataConstants

def OctoberEEGDataLoad(event):

    assert type(event) == int or type(event) == str, "The 'event' must be the event name as string or event number as integer"

    if type(event) == str:

        tmp = [LocalDataConstants.names['events'][i] == event for i in range(len(LocalDataConstants.names['events']))]

        assert np.any(tmp), "The Event is not available"

        event_number = np.where(tmp)[0][0]

    else:

        assert event >= 0 and event < len(LocalDataConstants.names['events']), "The Event is not available"
        
        event_number = event
        event = LocalDataConstants.names['events'][event_number]

    eeg_file_dir = LocalDataConstants.directories['eeg_file_dir']['October'][event]

    try:

        f = loadmat(eeg_file_dir)

        raw_data = f['All_data_' + str(LocalDataConstants.names['events'][event_number])]

        raw_data = raw_data.transpose(0, 2, 1, 3)

    except:

        with h5py.File(eeg_file_dir, 'r') as f:

            raw_data = f['All_data_' + str(LocalDataConstants.names['events'][event_number])][:]

        raw_data = raw_data.transpose(3, 1, 2, 0)

    try:

        DataLengthsDir = LocalDataConstants.directories['DataLengthsDir']

        f = loadmat(DataLengthsDir)
        data_lengths = f['data_lengths'][:, event_number]

    except:

        if event == 'Actions':

            DataLengthsDir = LocalDataConstants.directories['ActionDataLengthsDir']

            f = loadmat(DataLengthsDir)
            data_lengths = f['data_lengths']

        else:

            DataLengthsDir = LocalDataConstants.directories[event + 'DataLengthsDir']

            f = loadmat(DataLengthsDir)
            data_lengths = f['data_lengths']

    return raw_data, data_lengths

def SeptemberEEGDataLoad(event):

    assert type(event) == int or type(event) == str, "The 'event' must be the event name as string or event number as integer"

    if type(event) == str:

        assert event != 'Actions', "Actions Data Is not Available Yet"

        tmp = [LocalDataConstants.names['events'][i] == event for i in range(len(LocalDataConstants.names['events']))]

        assert np.any(tmp), "The Event is not available"

        event_number = np.where(tmp)[0][0]

    else:

        assert event >= 0 and event < len(LocalDataConstants.names['events']), "The Event is not available"
        assert event != 4, "Actions Data Is not Available Yet"
        
        event_number = event
        event = LocalDataConstants.names['events'][event_number]

    eeg_file_dir = LocalDataConstants.directories['eeg_file_dir']['September'][event]

    with h5py.File(eeg_file_dir, 'r') as f:

        raw_data = f['All_data_' + str(LocalDataConstants.names['events'][event_number])][:]

    raw_data = raw_data.transpose(3, 1, 2, 0)

    DataLengthsDir = LocalDataConstants.directories['DataLengthsDir']

    f = loadmat(DataLengthsDir)
    data_lengths = f['data_lengths'][:, event_number]

    return raw_data, data_lengths

def JulyEEGDataLoad(event):

    if event == 'Actions' or event == 4:

        EvGr = event

    else:

        EvGr = 'Others'

    eeg_file_dir = LocalDataConstants.directories['eeg_file_dir']['July'][EvGr]

    assert type(event) == int or type(event) == str, "The 'event' must be the event name as string or event number as integer"

    if type(event) == str:

        tmp = [LocalDataConstants.names['events'][i] == event for i in range(len(LocalDataConstants.names['events']))]

        assert np.any(tmp), "The Event is not available"

        event_number = np.where(tmp)[0][0]

    else:

        assert event >= 0 and event < len(LocalDataConstants.names['events']), "The Event is not available"
        
        event_number = event

    try:

        with h5py.File(eeg_file_dir, 'r') as f:

            raw_data = f['All_data_' + str(LocalDataConstants.names['events'][event_number])][:]

        with h5py.File(eeg_file_dir, 'r') as f:

            data_lengths = f['data_lengths'][event_number, :]

        raw_data = raw_data.transpose(3, 1, 2, 0)

    except:

        with h5py.File(eeg_file_dir['dir'], 'r') as f:

                raw_data = f['All_data_' + event]

        raw_data = raw_data.transpose(3, 1, 2, 0)

        DataLengthsDir = LocalDataConstants.directories['ActionDataLengthsDir']

        f = loadmat(DataLengthsDir)
        data_lengths = f['data_lengths']

    return raw_data, data_lengths