import numpy as np

class DC_Constants():

    Properties = {

        'PLI':{

            'directed': False,
            'lagged': False
        },

        'LRB_GC':{
            
            'directed': True,
            'lagged': True

        },

        'PIB_GC':{
            
            'directed': True,
            'lagged': True

        },

        'PAC':{
            
            'directed': True,
            'lagged': False

        },

        'TransferEntropy':{
            
            'directed': True,
            'lagged': True

        }
    }

class LocalDataConstants():

    directories = {

        'eeg_file_dir': r'E:\HWs\Msc\Research\Research\Depression Dataset\New Datasets\Clustered_SingleTrialData_All_Neg_Pos_Stim.mat',
        'beh_dir_file': r'E:\HWs\Msc\Research\Research\Depression Dataset\depression_rl_eeg\Depression PS Task\Scripts from Manuscript\Data_4_Import.xlsx',
        'perform_data_dir': r'E:\HWs\Msc\Research\Research\Depression Dataset\New Datasets\Subjects_Behavioral_datas.csv',
        'eeg_prep_datasets_dir': r'E:\\HWs\Msc\\Research\\Research\\Depression Dataset\\Testing Preprocess'
    }

    names = {

        'JulyClusterNames': ['PF', 'LF', 'RF', 'MFC', 'LT', 'RT', 'LFC', 'RFC', 'MPC', 'LPC', 'RPC', 'MP', 'LPO', 'RPO'],
        'freq_bands': ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'],
        'events': ['All', 'Neg', 'Pos', 'Stim']
    }

class SpectralConstants():

    BandsBounds = {

        'Delta': [0.5, 4],
        'Theta': [4, 8],
        'Alpha': [8, 12],
        'Beta': [12, 30],
        'Gamma': [30, 50]

    }